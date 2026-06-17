"""
Fluxograma resumido:
Entrada (frame da camera) -> Validacao (detecao/encoding e thresholds) -> Lógica (classificacao multi-face, ambiguidade, focus zone, historico) -> Saida (overlay OpenCV + APIs Flask).

Versao 1.2 - 2026-03-18 / Mudanca: Refatoracao para multiplas faces e historico.
"""

import os
import re
import json
import time
import shutil
import sqlite3
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, jsonify, request

APP_TITLE = "Face.ID Premium Web"
DB_DIR = "faces_db"
DB_NAMES = os.path.join(DB_DIR, "nomes.json")
DB_ENCODINGS = os.path.join(DB_DIR, "encodings.npy")
REFERENCE_DB_FILE = os.path.join(DB_DIR, "references.sqlite3")
REFERENCE_IMAGES_DIR = os.path.join(DB_DIR, "reference_images")
SETTINGS_FILE = os.path.join(DB_DIR, "settings.json")
HISTORY_FILE = "capture_history.jsonl"
ERROR_LOG_FILE = "error.log"

CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

ANALYSIS_SCALE = 0.5
ANALYSIS_INTERVAL = 0.14

SAMPLES_NEEDED = 5
SAMPLE_INTERVAL = 1.0
REGISTER_TIMEOUT_SEC = 40.0

TOLERANCE = 0.48
UNCERTAINTY_MARGIN = 0.03
CONFIRMED_SCORE_THRESHOLD = 0.199

FOCUS_ZONE_WIDTH_RATIO = 0.30
FOCUS_ZONE_HEIGHT_RATIO = 0.40

HISTORY_DEDUP_WINDOW_SEC = 2.0
SIGHTING_MIN_INTERVAL_SEC = 60.0

STATUS_IDLE = "Aguardando rosto na camera"
STATUS_ANALYZING = "Analisando..."
STATUS_AMBIGUOUS = "Ambiguidade detectada - use a zona de foco central"
STATUS_COLLECTING = "Coletando amostras..."
STATUS_RECOGNIZED = "Identificado"
STATUS_NOT_RECOGNIZED = "Desconhecido"
STATUS_MULTI_FACE = "Problema: mais de uma pessoa na camera"
STATUS_DB_EMPTY = "Base vazia"
STATUS_CAMERA_ERROR = "Erro de camera"

COLOR_IDLE = (180, 180, 180)
COLOR_ANALYZING = (0, 255, 255)  # amarelo
COLOR_OK = (0, 255, 0)  # verde
COLOR_FAIL = (0, 0, 255)  # vermelho


def log_error(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class FaceMatchResult:
    face_id: str
    location: tuple[int, int, int, int]  # (top, right, bottom, left)
    name: str
    distance: float | None
    matched: bool
    uncertain: bool
    inside_focus_zone: bool


class SettingsRepository:
    def __init__(self, file_path=SETTINGS_FILE):
        self.file_path = file_path
        self.lock = threading.Lock()
        os.makedirs(DB_DIR, exist_ok=True)
        self.defaults = {
            "confirmed_score_threshold": CONFIRMED_SCORE_THRESHOLD,
            "samples_needed": SAMPLES_NEEDED,
            "max_reference_photos_per_person": 15,
        }
        self.settings = self._load()

    def _sanitize(self, raw: dict):
        data = dict(self.defaults)
        if isinstance(raw, dict):
            data.update(raw)

        try:
            threshold = float(data.get("confirmed_score_threshold", CONFIRMED_SCORE_THRESHOLD))
        except (TypeError, ValueError):
            threshold = CONFIRMED_SCORE_THRESHOLD
        threshold = max(0.05, min(0.99, threshold))

        try:
            samples_needed = int(data.get("samples_needed", SAMPLES_NEEDED))
        except (TypeError, ValueError):
            samples_needed = SAMPLES_NEEDED
        samples_needed = max(2, min(20, samples_needed))

        try:
            max_refs = int(data.get("max_reference_photos_per_person", 15))
        except (TypeError, ValueError):
            max_refs = 15
        max_refs = max(1, min(60, max_refs))

        return {
            "confirmed_score_threshold": threshold,
            "samples_needed": samples_needed,
            "max_reference_photos_per_person": max_refs,
        }

    def _save_locked(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=2)

    def _load(self):
        loaded = {}
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    if isinstance(payload, dict):
                        loaded = payload
            except Exception:
                loaded = {}
        sanitized = self._sanitize(loaded)
        self.settings = sanitized
        self._save_locked()
        return sanitized

    def get_settings(self):
        with self.lock:
            return dict(self.settings)

    def update_settings(self, updates: dict):
        with self.lock:
            merged = dict(self.settings)
            if isinstance(updates, dict):
                merged.update(updates)
            self.settings = self._sanitize(merged)
            self._save_locked()
            return dict(self.settings)


class FaceRepository:
    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)
        self.names = []
        self.encodings = np.empty((0, 128), dtype=np.float64)
        self.load()

    def load(self):
        self.names = []
        self.encodings = np.empty((0, 128), dtype=np.float64)

        if os.path.exists(DB_NAMES):
            try:
                with open(DB_NAMES, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.names = data
            except Exception:
                self.names = []

        if os.path.exists(DB_ENCODINGS):
            try:
                if os.path.getsize(DB_ENCODINGS) > 0:
                    arr = np.load(DB_ENCODINGS, allow_pickle=False)
                    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 128:
                        self.encodings = arr.astype(np.float64)
                    else:
                        self.encodings = np.empty((0, 128), dtype=np.float64)
                else:
                    self.encodings = np.empty((0, 128), dtype=np.float64)
            except Exception:
                self.encodings = np.empty((0, 128), dtype=np.float64)

        if len(self.names) != len(self.encodings):
            self.names = []
            self.encodings = np.empty((0, 128), dtype=np.float64)
            self.save()

    def save(self):
        with open(DB_NAMES, "w", encoding="utf-8") as f:
            json.dump(self.names, f, ensure_ascii=False, indent=2)
        np.save(DB_ENCODINGS, self.encodings)

    def add_face(self, name: str, encoding: np.ndarray):
        clean_name = (name or "").strip()
        if not clean_name:
            return False
        self.names.append(clean_name)
        if self.encodings.shape[0] == 0:
            self.encodings = np.array([encoding], dtype=np.float64)
        else:
            self.encodings = np.vstack([self.encodings, encoding.astype(np.float64)])
        self.save()
        return True

    def get_registered_users_count(self):
        return len(set(self.names))

    def get_total_samples(self):
        return len(self.names)

    def get_sample_counts_by_person(self):
        counts = {}
        for name in self.names:
            counts[name] = counts.get(name, 0) + 1
        return counts

    def delete_person(self, person_name: str):
        target = (person_name or "").strip()
        if not target:
            return 0
        keep_indices = [idx for idx, name in enumerate(self.names) if name != target]
        removed = len(self.names) - len(keep_indices)
        if removed <= 0:
            return 0
        self.names = [self.names[idx] for idx in keep_indices]
        if keep_indices:
            self.encodings = self.encodings[keep_indices, :]
        else:
            self.encodings = np.empty((0, 128), dtype=np.float64)
        self.save()
        return removed

    def clear_all(self):
        removed = len(self.names)
        self.names = []
        self.encodings = np.empty((0, 128), dtype=np.float64)
        self.save()
        return removed

    def classify_encoding(
        self,
        encoding: np.ndarray,
        tolerance: float = TOLERANCE,
        uncertainty_margin: float = UNCERTAINTY_MARGIN,
        confirmed_score_threshold: float = CONFIRMED_SCORE_THRESHOLD,
    ):
        if len(self.names) == 0 or self.encodings.shape[0] == 0:
            return "Desconhecido", None, False, True

        distances = face_recognition.face_distance(self.encodings, encoding)
        sorted_idx = np.argsort(distances)
        best_idx = int(sorted_idx[0])
        best_distance = float(distances[best_idx])
        best_score = max(0.0, 1.0 - best_distance)

        second_distance = None
        if len(sorted_idx) > 1:
            second_distance = float(distances[int(sorted_idx[1])])

        matched = best_score >= float(confirmed_score_threshold)
        close_competition = (
            second_distance is not None and abs(best_distance - second_distance) < uncertainty_margin
        )
        uncertain = (not matched) or close_competition
        name = self.names[best_idx] if matched else "Desconhecido"
        return name, best_distance, matched, uncertain


class ReferenceImageStore:
    def __init__(self, db_path=REFERENCE_DB_FILE, images_dir=REFERENCE_IMAGES_DIR):
        self.db_path = db_path
        self.images_dir = images_dir
        os.makedirs(DB_DIR, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        self._ensure_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reference_person ON reference_images(person_name)"
            )

    def _safe_person_name(self, person_name: str):
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (person_name or "").strip())
        return cleaned[:80] if cleaned else "sem_nome"

    def save_reference(self, person_name: str, frame_bgr: np.ndarray, face_box):
        top, right, bottom, left = face_box
        face_h = max(1, bottom - top)
        face_w = max(1, right - left)
        margin_y = int(face_h * 0.25)
        margin_x = int(face_w * 0.25)

        h, w = frame_bgr.shape[:2]
        y1 = max(0, top - margin_y)
        y2 = min(h, bottom + margin_y)
        x1 = max(0, left - margin_x)
        x2 = min(w, right + margin_x)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        person_dir = os.path.join(self.images_dir, self._safe_person_name(person_name))
        os.makedirs(person_dir, exist_ok=True)

        file_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        image_path = os.path.join(person_dir, file_name)
        if not cv2.imwrite(image_path, crop):
            return None

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO reference_images (person_name, image_path, created_at) VALUES (?, ?, ?)",
                (person_name, image_path, now_iso()),
            )
        return image_path

    def count_references(self):
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM reference_images").fetchone()
            return int(row[0]) if row else 0

    def get_reference_counts_by_person(self):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT person_name, COUNT(*) FROM reference_images GROUP BY person_name"
            ).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def count_person_references(self, person_name: str):
        target = (person_name or "").strip()
        if not target:
            return 0
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM reference_images WHERE person_name = ?",
                (target,),
            ).fetchone()
        return int(row[0]) if row else 0

    def trim_person_references(self, person_name: str, max_count: int):
        target = (person_name or "").strip()
        if not target:
            return 0
        limit = max(1, int(max_count))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, image_path
                FROM reference_images
                WHERE person_name = ?
                ORDER BY created_at DESC, id DESC
                """,
                (target,),
            ).fetchall()
            if len(rows) <= limit:
                return 0
            to_remove = rows[limit:]
            conn.executemany("DELETE FROM reference_images WHERE id = ?", [(row[0],) for row in to_remove])

        removed = 0
        for _, image_path in to_remove:
            try:
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                removed += 1
            except Exception:
                log_error(f"Falha ao remover referencia excedente: {image_path}")
        return removed

    def delete_person_references(self, person_name: str):
        target = (person_name or "").strip()
        if not target:
            return 0

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT image_path FROM reference_images WHERE person_name = ?",
                (target,),
            ).fetchall()
            conn.execute("DELETE FROM reference_images WHERE person_name = ?", (target,))

        removed = 0
        for row in rows:
            img_path = row[0]
            try:
                if img_path and os.path.exists(img_path):
                    os.remove(img_path)
                removed += 1
            except Exception:
                log_error(f"Falha ao remover foto de referencia: {img_path}")

        person_dir = os.path.join(self.images_dir, self._safe_person_name(target))
        try:
            if os.path.isdir(person_dir):
                shutil.rmtree(person_dir, ignore_errors=True)
        except Exception:
            log_error(f"Falha ao remover pasta de referencia: {person_dir}")
        return removed

    def clear_all_references(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM reference_images")
        try:
            if os.path.isdir(self.images_dir):
                shutil.rmtree(self.images_dir, ignore_errors=True)
            os.makedirs(self.images_dir, exist_ok=True)
        except Exception:
            log_error("Falha ao limpar diretorio de fotos de referencia.")


class HistoryRepository:
    def __init__(self, file_path=HISTORY_FILE, dedup_window_sec=HISTORY_DEDUP_WINDOW_SEC):
        self.file_path = file_path
        self.dedup_window_sec = dedup_window_sec
        self.lock = threading.Lock()
        self.recent_buffer = {}  # key: (camera_id, pessoas) -> timestamp
        self.last_sighting_by_person = {}  # key: (camera_id, person_name) -> timestamp
        self.sightings_cache = []
        self.max_cached_sightings = 300
        if not os.path.exists(self.file_path):
            with open(self.file_path, "a", encoding="utf-8"):
                pass
        self._warmup_recent_sightings()

    def _make_people_signature(self, results: list[FaceMatchResult]):
        people = [r.name if r.matched else "Desconhecido" for r in results]
        return tuple(sorted(people))

    def _cleanup_buffer(self, now_ts: float):
        stale = []
        for key, ts in self.recent_buffer.items():
            if now_ts - ts > self.dedup_window_sec:
                stale.append(key)
        for key in stale:
            self.recent_buffer.pop(key, None)

    def _to_epoch(self, timestamp_value):
        if not timestamp_value:
            return None
        try:
            return datetime.fromisoformat(str(timestamp_value)).timestamp()
        except Exception:
            return None

    def _cache_sighting(self, payload: dict):
        self.sightings_cache.append(
            {
                "timestamp": payload.get("timestamp"),
                "camera_id": payload.get("camera_id"),
                "person_name": payload.get("person_name"),
                "distance": payload.get("distance"),
                "face_id": payload.get("face_id"),
                "event_type": payload.get("event_type"),
            }
        )
        if len(self.sightings_cache) > self.max_cached_sightings:
            self.sightings_cache = self.sightings_cache[-self.max_cached_sightings :]

    def _warmup_recent_sightings(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue

                    if payload.get("event_type") != "sighting_confirmed":
                        continue

                    camera_id = str(payload.get("camera_id", CAMERA_INDEX))
                    person_name = str(payload.get("person_name", "")).strip()
                    ts_epoch = self._to_epoch(payload.get("timestamp"))
                    if person_name and ts_epoch is not None:
                        key = (camera_id, person_name)
                        prev = self.last_sighting_by_person.get(key)
                        if prev is None or ts_epoch > prev:
                            self.last_sighting_by_person[key] = ts_epoch
                    self._cache_sighting(payload)
        except Exception:
            log_error("Falha ao carregar historico de avistamentos existente.")

    def append_event(self, camera_id: str, event_type: str, results: list[FaceMatchResult], focus_zone_active: bool):
        now_ts = time.time()
        people_signature = self._make_people_signature(results)
        dedup_key = (str(camera_id), people_signature)

        with self.lock:
            self._cleanup_buffer(now_ts)
            last_seen = self.recent_buffer.get(dedup_key)
            if last_seen is not None and (now_ts - last_seen) < self.dedup_window_sec:
                return False
            self.recent_buffer[dedup_key] = now_ts

            payload = {
                "timestamp": now_iso(),
                "camera_id": str(camera_id),
                "event_type": event_type,
                "focus_zone_active": bool(focus_zone_active),
                "faces_count": len(results),
                "people": list(people_signature),
                "faces": [
                    {
                        "face_id": r.face_id,
                        "location": list(r.location),
                        "name": r.name,
                        "distance": round(r.distance, 4) if r.distance is not None else None,
                        "matched": r.matched,
                        "uncertain": r.uncertain,
                        "inside_focus_zone": r.inside_focus_zone,
                    }
                    for r in results
                ],
            }
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True

    def append_sightings(self, camera_id: str, results: list[FaceMatchResult], min_interval_sec: float = SIGHTING_MIN_INTERVAL_SEC):
        now_ts = time.time()
        timestamp = now_iso()
        payloads = []

        with self.lock:
            for result in results:
                if not result.matched or result.uncertain:
                    continue

                person_name = (result.name or "").strip()
                if not person_name or person_name.lower() == "desconhecido":
                    continue

                sighting_key = (str(camera_id), person_name)
                last_seen = self.last_sighting_by_person.get(sighting_key)
                if last_seen is not None and (now_ts - last_seen) < float(min_interval_sec):
                    continue

                self.last_sighting_by_person[sighting_key] = now_ts
                payloads.append(
                    {
                        "timestamp": timestamp,
                        "camera_id": str(camera_id),
                        "event_type": "sighting_confirmed",
                        "person_name": person_name,
                        "distance": round(result.distance, 4) if result.distance is not None else None,
                        "face_id": result.face_id,
                    }
                )

            if not payloads:
                return 0

            with open(self.file_path, "a", encoding="utf-8") as f:
                for payload in payloads:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    self._cache_sighting(payload)
        return len(payloads)

    def get_recent_sightings(self, limit=30):
        try:
            safe_limit = max(1, min(200, int(limit)))
        except (TypeError, ValueError):
            safe_limit = 30

        with self.lock:
            if not self.sightings_cache:
                return []
            return list(reversed(self.sightings_cache[-safe_limit:]))

    def clear_sightings(self):
        removed = 0
        kept_lines = []
        with self.lock:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            kept_lines.append(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                            continue

                        if payload.get("event_type") == "sighting_confirmed":
                            removed += 1
                            continue
                        kept_lines.append(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                return 0

            with open(self.file_path, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
            self.last_sighting_by_person = {}
            self.sightings_cache = []
            self.recent_buffer = {}
        return removed


class CameraWorker:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            log_error("Nao foi possivel abrir a webcam.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            try:
                ok, frame = self.cap.read()
                if ok:
                    with self.frame_lock:
                        self.current_frame = frame
                time.sleep(0.01)
            except Exception:
                log_error(f"Erro no capture_loop:\n{traceback.format_exc()}")
                time.sleep(0.1)

    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()


class WebTracker:
    def __init__(self):
        self.repo = FaceRepository()
        self.reference_store = ReferenceImageStore()
        self.history_repo = HistoryRepository()
        self.settings_repo = SettingsRepository()
        self.camera = CameraWorker(CAMERA_INDEX)

        self.data_lock = threading.Lock()
        self.lock = threading.Lock()

        self.status_text = STATUS_IDLE
        self.last_name = "-"
        self.last_distance = "-"
        self.lbl_users_count = str(self.repo.get_registered_users_count())
        self.lbl_samples_count = str(self.repo.get_total_samples())
        self.reference_count = self.reference_store.count_references()
        self.lbl_reference_count = str(self.reference_count)

        self.pending_action = None
        self.processing = False
        self.samples_collected = 0
        self.last_sample_time = 0.0
        self.action_started_at = 0.0
        self.last_action_analysis_time = 0.0

        self.last_live_analysis_time = 0.0
        self.current_matches: list[FaceMatchResult] = []
        self.focus_zone_active = False
        self.focus_zone_rect = None

        self.result_color = COLOR_IDLE
        self.result_label = STATUS_IDLE
        self.output_frame = None
        self.confirmed_score_threshold = CONFIRMED_SCORE_THRESHOLD
        self.samples_needed = SAMPLES_NEEDED
        self.max_reference_photos_per_person = 15
        self._apply_settings(self.settings_repo.get_settings())

        self.camera.start()
        if not self.camera.running:
            self.set_status("Webcam indisponivel.", COLOR_FAIL, STATUS_CAMERA_ERROR, name="Erro", distance="-")
        threading.Thread(target=self.process_video_stream, daemon=True).start()

    def _apply_settings(self, settings: dict):
        self.confirmed_score_threshold = float(settings.get("confirmed_score_threshold", CONFIRMED_SCORE_THRESHOLD))
        self.samples_needed = int(settings.get("samples_needed", SAMPLES_NEEDED))
        self.max_reference_photos_per_person = int(settings.get("max_reference_photos_per_person", 15))

    def get_settings(self):
        return {
            "confirmed_score_threshold": round(float(self.confirmed_score_threshold), 3),
            "samples_needed": int(self.samples_needed),
            "max_reference_photos_per_person": int(self.max_reference_photos_per_person),
        }

    def update_settings(self, updates: dict):
        saved = self.settings_repo.update_settings(updates)
        self._apply_settings(saved)
        return self.get_settings()

    def is_collecting(self):
        return bool(self.processing and self.pending_action and self.pending_action[0] == "register")

    def get_collection_instructions(self):
        if self.is_collecting():
            return (
                f"Coleta ativa ({self.samples_collected}/{self.samples_needed}): fique sozinho na camera, "
                "mantenha o rosto centralizado no quadrado verde, olhe para frente e mude levemente "
                "o angulo da cabeca a cada nova amostra."
            )
        return (
            "Coleta correta: mantenha boa iluminacao, um rosto por vez e fique centralizado na camera."
        )

    def _sync_counters(self):
        with self.data_lock:
            self.lbl_users_count = str(self.repo.get_registered_users_count())
            self.lbl_samples_count = str(self.repo.get_total_samples())
            self.reference_count = self.reference_store.count_references()
        self.lbl_reference_count = str(self.reference_count)

    def set_status(self, text, color=None, label=None, name=None, distance=None):
        self.status_text = text
        if name is not None:
            self.last_name = name
        if distance is not None:
            self.last_distance = distance
        if color is not None:
            self.result_color = color
        if label is not None:
            self.result_label = label
        self._sync_counters()

    def _reset_processing_state(self):
        self.processing = False
        self.pending_action = None
        self.samples_collected = 0
        self.last_sample_time = 0.0
        self.action_started_at = 0.0
        self.last_action_analysis_time = 0.0

    def reset_state(self):
        self._reset_processing_state()
        self.current_matches = []
        self.focus_zone_active = False
        self.focus_zone_rect = None
        self.set_status("Aguardando rosto na camera.", COLOR_IDLE, STATUS_IDLE, name="-", distance="-")

    def trigger_register(self, name):
        if self.processing:
            return

        clean_name = (name or "").strip()
        if not clean_name:
            self.set_status("Informe um nome valido para cadastro.", COLOR_FAIL, STATUS_IDLE)
            return

        self.pending_action = ("register", clean_name)
        self.processing = True
        self.samples_collected = 0
        self.last_sample_time = time.time() - SAMPLE_INTERVAL
        self.action_started_at = time.time()
        self.last_action_analysis_time = 0.0
        self.current_matches = []
        self.focus_zone_active = False
        self.focus_zone_rect = None
        self.set_status(
            f"Preparando cadastro: coletando {self.samples_needed} amostras...",
            COLOR_OK,
            STATUS_COLLECTING,
            name=clean_name,
            distance="-",
        )

    def trigger_validate(self):
        self.set_status(
            "Reconhecimento multiplo automatico ativo no video principal.",
            COLOR_ANALYZING,
            STATUS_ANALYZING,
        )

    def _scale_box(self, box, frame_shape):
        top, right, bottom, left = box
        inv = 1.0 / ANALYSIS_SCALE
        top = int(top * inv)
        right = int(right * inv)
        bottom = int(bottom * inv)
        left = int(left * inv)

        h, w = frame_shape[:2]
        top = max(0, min(top, h - 1))
        right = max(0, min(right, w - 1))
        bottom = max(0, min(bottom, h - 1))
        left = max(0, min(left, w - 1))
        return (top, right, bottom, left)

    def _detect_faces_and_encodings(self, frame):
        small = cv2.resize(frame, (0, 0), fx=ANALYSIS_SCALE, fy=ANALYSIS_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, known_face_locations=locations) if locations else []
        return locations, encodings

    def _compute_focus_zone(self, frame_shape):
        h, w = frame_shape[:2]
        zone_w = int(w * FOCUS_ZONE_WIDTH_RATIO)
        zone_h = int(h * FOCUS_ZONE_HEIGHT_RATIO)
        left = (w - zone_w) // 2
        top = (h - zone_h) // 2
        right = left + zone_w
        bottom = top + zone_h
        return (left, top, right, bottom)

    def _inside_focus_zone(self, face_location, focus_zone):
        top, right, bottom, left = face_location
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        z_left, z_top, z_right, z_bottom = focus_zone
        return z_left <= cx <= z_right and z_top <= cy <= z_bottom

    def _map_results(self, locations_small, encodings, frame_shape):
        results = []
        for idx, box_small in enumerate(locations_small):
            full_box = self._scale_box(box_small, frame_shape)
            if idx < len(encodings):
                with self.data_lock:
                    name, distance, matched, uncertain = self.repo.classify_encoding(
                        encodings[idx], TOLERANCE, UNCERTAINTY_MARGIN, self.confirmed_score_threshold
                    )
            else:
                name, distance, matched, uncertain = ("Desconhecido", None, False, True)

            results.append(
                FaceMatchResult(
                    face_id=f"face_{idx}",
                    location=full_box,
                    name=name,
                    distance=distance,
                    matched=matched,
                    uncertain=uncertain,
                    inside_focus_zone=False,
                )
            )
        return results

    def _get_event_type(self, results: list[FaceMatchResult]):
        if not results:
            return "no_face"
        if len(results) >= 2:
            if any(r.uncertain for r in results):
                return "ambiguous_multiple"
            if all(r.matched for r in results):
                return "identified_multiple"
            return "unknown_multiple"
        first = results[0]
        if not first.matched:
            return "unknown_single"
        if first.uncertain:
            return "ambiguous_single"
        return "identified_single"

    def _apply_focus_overlay(self, frame, focus_zone):
        dark = np.zeros_like(frame)
        dimmed = cv2.addWeighted(frame, 0.35, dark, 0.65, 0.0)
        left, top, right, bottom = focus_zone
        dimmed[top:bottom, left:right] = frame[top:bottom, left:right]
        cv2.rectangle(dimmed, (left, top), (right, bottom), COLOR_ANALYZING, 2)
        cv2.putText(
            dimmed,
            "Zona de Foco",
            (left + 8, max(20, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_ANALYZING,
            2,
            cv2.LINE_AA,
        )
        return dimmed

    def _get_result_color(self, result: FaceMatchResult):
        if self.is_collecting():
            if len(self.current_matches) == 1:
                return COLOR_OK
            return COLOR_FAIL
        if not result.matched:
            return COLOR_FAIL
        if result.uncertain:
            return COLOR_ANALYZING
        return COLOR_OK

    def _result_label(self, result: FaceMatchResult):
        if self.is_collecting():
            if len(self.current_matches) == 1:
                return f"Coleta {self.samples_collected}/{self.samples_needed}"
            return "Cadastro: mantenha apenas 1 pessoa"

        dist = f"{result.distance:.4f}" if result.distance is not None else "-"
        if not result.matched:
            label = f"Desconhecido ({dist})"
        elif result.uncertain:
            label = f"{result.name} ? ({dist})"
        else:
            label = f"{result.name} ({dist})"

        if self.focus_zone_active:
            suffix = "Foco" if result.inside_focus_zone else "Fora do foco"
            label = f"{label} - {suffix}"
        return label

    def _draw_status_banner(self, frame, text, color):
        if not text:
            return
        text_color = (255, 255, 255) if color == COLOR_FAIL else (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x1, y1 = 12, 12
        x2, y2 = min(frame.shape[1] - 12, x1 + tw + 22), y1 + th + 18
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, text, (x1 + 10, y2 - 8), font, scale, text_color, thickness, cv2.LINE_AA)

    def _save_reference_photo(self, person_name, frame, face_box):
        with self.data_lock:
            path = self.reference_store.save_reference(person_name, frame, face_box)
            if path:
                self.reference_store.trim_person_references(person_name, self.max_reference_photos_per_person)
            self.reference_count = self.reference_store.count_references()
        if not path:
            log_error(f"Falha ao salvar referencia para {person_name}.")

    def _analyze_register(self, frame, now):
        if not self.processing or not self.pending_action:
            return
        _, register_name = self.pending_action

        if (now - self.action_started_at) > REGISTER_TIMEOUT_SEC:
            self._reset_processing_state()
            self.set_status("Tempo excedido no cadastro. Tente novamente.", COLOR_FAIL, STATUS_IDLE)
            return

        locations_small, encodings = self._detect_faces_and_encodings(frame)
        if len(locations_small) == 0:
            self.current_matches = []
            self.set_status(
                "Coleta: aproxime e centralize 1 rosto no quadrado verde.",
                COLOR_ANALYZING,
                STATUS_COLLECTING,
                name=register_name,
            )
            return

        if len(locations_small) > 1:
            results = self._map_results(locations_small, encodings, frame.shape)
            self.current_matches = results
            self.focus_zone_active = False
            self.focus_zone_rect = None
            self.set_status("Coleta pausada: deixe apenas 1 pessoa na camera.", COLOR_FAIL, STATUS_MULTI_FACE, name=register_name)
            return

        results = self._map_results(locations_small, encodings, frame.shape)
        self.current_matches = results
        self.focus_zone_active = False
        self.focus_zone_rect = None

        if now - self.last_sample_time < SAMPLE_INTERVAL:
            self.set_status(
                f"Coleta ativa {self.samples_collected}/{self.samples_needed}: mantenha firme e mude levemente o angulo.",
                COLOR_OK,
                STATUS_COLLECTING,
                name=register_name,
            )
            return

        if not encodings:
            self.set_status("Coleta: ajuste iluminacao e mantenha o rosto frontal.", COLOR_ANALYZING, STATUS_COLLECTING)
            return

        with self.data_lock:
            added = self.repo.add_face(register_name, encodings[0])

        if added:
            self.samples_collected += 1
            self.last_sample_time = now
            self._save_reference_photo(register_name, frame, results[0].location)
            self.set_status(
                f"Amostra {self.samples_collected}/{self.samples_needed} capturada. Mude levemente o angulo para a proxima.",
                COLOR_OK,
                STATUS_COLLECTING,
                name=register_name,
                distance="-",
            )

        if self.samples_collected >= self.samples_needed:
            self._reset_processing_state()
            self.set_status(
                f"Cadastro concluido: {register_name}. Reconhecimento multiplo ativo.",
                COLOR_OK,
                STATUS_RECOGNIZED,
                name=register_name,
                distance="-",
            )

    def _analyze_live(self, frame):
        with self.data_lock:
            has_people = len(self.repo.names) > 0
        if not has_people:
            self.current_matches = []
            self.focus_zone_active = False
            self.focus_zone_rect = None
            self.set_status("Cadastre a primeira pessoa para iniciar reconhecimento.", COLOR_IDLE, STATUS_DB_EMPTY, name="-", distance="-")
            return

        locations_small, encodings = self._detect_faces_and_encodings(frame)
        results = self._map_results(locations_small, encodings, frame.shape)

        focus_active = len(results) >= 2 and any(r.uncertain for r in results)
        focus_rect = self._compute_focus_zone(frame.shape) if focus_active else None
        if focus_active and focus_rect is not None:
            updated = []
            for r in results:
                r.inside_focus_zone = self._inside_focus_zone(r.location, focus_rect)
                updated.append(r)
            results = updated

        self.current_matches = results
        self.focus_zone_active = focus_active
        self.focus_zone_rect = focus_rect

        event_type = self._get_event_type(results)
        try:
            self.history_repo.append_event(str(CAMERA_INDEX), event_type, results, focus_active)
            self.history_repo.append_sightings(str(CAMERA_INDEX), results, SIGHTING_MIN_INTERVAL_SEC)
        except Exception:
            log_error(f"Falha ao gravar historico:\n{traceback.format_exc()}")

        if not results:
            self.set_status("Aguardando rosto na camera...", COLOR_IDLE, STATUS_IDLE, name="-", distance="-")
            return

        if focus_active:
            inside_count = sum(1 for r in results if r.inside_focus_zone)
            if inside_count == 0:
                self.set_status(
                    "Ambiguidade: posicione uma pessoa na zona de foco central.",
                    COLOR_ANALYZING,
                    STATUS_AMBIGUOUS,
                    name="-",
                    distance="-",
                )
            else:
                self.set_status(
                    "Ambiguidade detectada: mantenha a pessoa dentro da zona de foco.",
                    COLOR_ANALYZING,
                    STATUS_AMBIGUOUS,
                    name="-",
                    distance="-",
                )
            return

        if len(results) == 1:
            result = results[0]
            if result.matched and not result.uncertain:
                self.set_status(
                    f"Identificado: {result.name}",
                    COLOR_OK,
                    STATUS_RECOGNIZED,
                    name=result.name,
                    distance=f"{result.distance:.4f}" if result.distance is not None else "-",
                )
            elif result.matched and result.uncertain:
                self.set_status(
                    f"Analisando semelhanca de {result.name}...",
                    COLOR_ANALYZING,
                    STATUS_ANALYZING,
                    name=result.name,
                    distance=f"{result.distance:.4f}" if result.distance is not None else "-",
                )
            else:
                self.set_status(
                    "Desconhecido",
                    COLOR_FAIL,
                    STATUS_NOT_RECOGNIZED,
                    name="Desconhecido",
                    distance=f"{result.distance:.4f}" if result.distance is not None else "-",
                )
            return

        known_count = sum(1 for r in results if r.matched and not r.uncertain)
        if known_count == len(results):
            names = ", ".join(r.name for r in results[:3])
            self.set_status(
                f"Multiplos identificados: {names}",
                COLOR_OK,
                STATUS_RECOGNIZED,
                name=names,
                distance="-",
            )
        else:
            self.set_status(
                STATUS_MULTI_FACE,
                COLOR_ANALYZING,
                STATUS_ANALYZING,
                name="-",
                distance="-",
            )

    def draw_overlay(self, frame):
        display = frame.copy()

        if self.focus_zone_active and self.focus_zone_rect is not None:
            display = self._apply_focus_overlay(display, self.focus_zone_rect)

        for result in self.current_matches:
            color = self._get_result_color(result)
            top, right, bottom, left = result.location
            cv2.rectangle(display, (left, top), (right, bottom), color, 2)

            label = self._result_label(result)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            y1 = max(0, top - 26)
            y2 = y1 + th + 10
            x2 = min(display.shape[1] - 1, left + tw + 12)
            cv2.rectangle(display, (left, y1), (x2, y2), color, -1)
            text_color = (255, 255, 255) if color == COLOR_FAIL else (0, 0, 0)
            cv2.putText(
                display,
                label,
                (left + 6, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                text_color,
                2,
                cv2.LINE_AA,
            )

        banner_color = COLOR_OK if self.is_collecting() else self.result_color
        banner_text = (
            f"{STATUS_COLLECTING} {self.samples_collected}/{self.samples_needed}"
            if self.is_collecting()
            else self.result_label
        )
        self._draw_status_banner(display, banner_text, banner_color)
        return display

    def process_video_stream(self):
        while True:
            try:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.03)
                    continue

                frame = cv2.flip(frame, 1)
                now = time.time()

                if self.processing and self.pending_action and self.pending_action[0] == "register":
                    if (now - self.last_action_analysis_time) >= ANALYSIS_INTERVAL:
                        try:
                            self._analyze_register(frame, now)
                        except Exception:
                            log_error(f"Erro no cadastro:\n{traceback.format_exc()}")
                            self._reset_processing_state()
                            self.current_matches = []
                            self.focus_zone_active = False
                            self.focus_zone_rect = None
                            self.set_status("Erro interno durante cadastro.", COLOR_FAIL, STATUS_NOT_RECOGNIZED, name="Erro", distance="-")
                        finally:
                            self.last_action_analysis_time = now
                else:
                    if (now - self.last_live_analysis_time) >= ANALYSIS_INTERVAL:
                        try:
                            self._analyze_live(frame)
                        except Exception:
                            log_error(f"Erro no reconhecimento multiplo:\n{traceback.format_exc()}")
                            self.current_matches = []
                            self.focus_zone_active = False
                            self.focus_zone_rect = None
                            self.set_status("Erro interno no reconhecimento.", COLOR_FAIL, STATUS_NOT_RECOGNIZED, name="Erro", distance="-")
                        finally:
                            self.last_live_analysis_time = now

                rendered = self.draw_overlay(frame)
                with self.lock:
                    self.output_frame = rendered.copy()
            except Exception:
                log_error(f"Erro no process_video_stream:\n{traceback.format_exc()}")
                time.sleep(0.1)
            time.sleep(0.03)

    def get_jpg_bytes(self):
        with self.lock:
            if self.output_frame is None:
                return None
            ret, buffer = cv2.imencode(".jpg", self.output_frame)
            return buffer.tobytes() if ret else None

    def get_registrations_overview(self):
        with self.data_lock:
            sample_counts = self.repo.get_sample_counts_by_person()
            reference_counts = self.reference_store.get_reference_counts_by_person()
        names = sorted(set(sample_counts.keys()) | set(reference_counts.keys()), key=lambda n: n.lower())
        rows = []
        for name in names:
            rows.append(
                {
                    "name": name,
                    "samples": int(sample_counts.get(name, 0)),
                    "references": int(reference_counts.get(name, 0)),
                }
            )
        return rows

    def get_recent_history(self, limit=30):
        return self.history_repo.get_recent_sightings(limit)

    def clear_history_sightings(self):
        return self.history_repo.clear_sightings()

    def delete_registration(self, person_name):
        target = (person_name or "").strip()
        if not target:
            return False, 0, 0

        with self.data_lock:
            removed_samples = self.repo.delete_person(target)
            removed_refs = self.reference_store.delete_person_references(target)
            self.reference_count = self.reference_store.count_references()
            total_samples = self.repo.get_total_samples()

        if removed_samples == 0 and removed_refs == 0:
            return False, 0, 0

        if self.processing and self.pending_action and self.pending_action[0] == "register":
            if self.pending_action[1] == target:
                self._reset_processing_state()

        self.current_matches = []
        self.focus_zone_active = False
        self.focus_zone_rect = None

        if total_samples == 0:
            self.set_status("Base vazia. Cadastre alguem para iniciar.", COLOR_IDLE, STATUS_DB_EMPTY, name="-", distance="-")
        else:
            self.set_status(f"Cadastro removido: {target}", COLOR_ANALYZING, STATUS_IDLE, name="-", distance="-")
        return True, removed_samples, removed_refs

    def clear_registrations(self):
        with self.data_lock:
            removed_samples = self.repo.clear_all()
            self.reference_store.clear_all_references()
            self.reference_count = self.reference_store.count_references()

        self._reset_processing_state()
        self.current_matches = []
        self.focus_zone_active = False
        self.focus_zone_rect = None
        self.set_status("Base de cadastros limpa.", COLOR_IDLE, STATUS_DB_EMPTY, name="-", distance="-")
        return removed_samples


app = Flask(__name__)
tracker = WebTracker()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/cadastros")
def cadastros():
    return render_template("cadastros.html")


def gen_frames():
    while True:
        frame = tracker.get_jpg_bytes()
        if frame is not None:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def status():
    return jsonify(
        {
            "status_text": tracker.status_text,
            "last_name": tracker.last_name,
            "last_distance": tracker.last_distance,
            "users_count": tracker.lbl_users_count,
            "samples_count": tracker.lbl_samples_count,
            "references_count": tracker.lbl_reference_count,
            "faces_count": len(tracker.current_matches),
            "focus_zone_active": tracker.focus_zone_active,
            "is_collecting": tracker.is_collecting(),
            "collect_samples": tracker.samples_collected,
            "collect_target": tracker.samples_needed,
            "collect_instructions": tracker.get_collection_instructions(),
        }
    )


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "GET":
        return jsonify(tracker.get_settings())

    data = request.get_json(silent=True) or {}
    settings_payload = tracker.update_settings(data)
    return jsonify({"success": True, "settings": settings_payload})


@app.route("/api/history")
def history():
    limit_raw = request.args.get("limit", "30")
    try:
        limit = max(1, min(200, int(limit_raw)))
    except (TypeError, ValueError):
        limit = 30
    return jsonify({"items": tracker.get_recent_history(limit)})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    removed = tracker.clear_history_sightings()
    return jsonify({"success": True, "removed": removed})


@app.route("/api/registrations")
def registrations():
    return jsonify({"items": tracker.get_registrations_overview()})


@app.route("/api/registration/delete", methods=["POST"])
def delete_registration():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "")
    ok, removed_samples, removed_refs = tracker.delete_registration(name)
    if not ok:
        return jsonify({"success": False, "error": "cadastro_nao_encontrado"}), 404
    return jsonify(
        {
            "success": True,
            "removed_samples": removed_samples,
            "removed_references": removed_refs,
        }
    )


@app.route("/api/registration/clear", methods=["POST"])
def clear_registration():
    removed_samples = tracker.clear_registrations()
    return jsonify({"success": True, "removed_samples": removed_samples})


@app.route("/api/action", methods=["POST"])
def action():
    data = request.get_json(silent=True) or {}
    act = data.get("action")
    if act == "register":
        tracker.trigger_register(data.get("name", ""))
        return jsonify({"success": True})
    if act == "validate":
        tracker.trigger_validate()
        return jsonify({"success": True})
    if act == "cancel":
        tracker.reset_state()
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "acao_invalida"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

