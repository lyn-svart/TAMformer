import json
import os
from collections import defaultdict

from motion_labels import MOTION_TO_CLASS


class CarMotion(object):
    def __init__(self, data_path='./data/car_motion'):
        self.data_path = data_path

    def generate_data_trajectory_sequence(self, split, **opts):
        annotation_file = self._annotation_file(split, opts)

        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        if self._is_frame_indexed(annotations):
            return self._from_frame_indexed_annotations(annotations, opts)

        records = self._records_for_split(annotations, split)
        return self._from_track_records(records, opts)

    def _from_track_records(self, records, opts):
        seq_len = int(opts.get('min_track_size', 10))
        stride = int(opts.get('fstride', 1))

        images = []
        bboxes = []
        pids = []
        labels = []
        track_ids = []

        for record in records:
            frame_paths, boxes = self._frames_and_boxes(record)
            label = self._label(record, opts)
            if label is None:
                continue
            track_id = str(record.get('track_id', record.get('id', len(track_ids))))
            if len(frame_paths) < seq_len:
                continue
            for start in range(0, len(frame_paths) - seq_len + 1, stride):
                end = start + seq_len
                images.append([self._resolve_path(p) for p in frame_paths[start:end]])
                bboxes.append(boxes[start:end])
                pids.append([[track_id] for _ in range(seq_len)])
                labels.append(label)
                track_ids.append(track_id)

        return {
            'image': images,
            'bbox': bboxes,
            'pid': pids,
            'track_id': track_ids,
            'label': labels,
        }

    def _from_frame_indexed_annotations(self, annotations, opts):
        seq_len = int(opts.get('min_track_size', 10))
        stride = int(opts.get('fstride', 1))
        allowed_types = opts.get('object_types')
        if isinstance(allowed_types, str):
            allowed_types = [allowed_types]
        allowed_types = set(allowed_types or [])
        ignored_types = opts.get('ignored_object_types', ['Human'])
        if isinstance(ignored_types, str):
            ignored_types = [ignored_types]
        ignored_types = set(ignored_types or [])

        tracks = defaultdict(list)
        for frame_path in sorted(annotations.keys(), key=self._frame_sort_key):
            frame_data = annotations[frame_path]
            drive_id = self._drive_id(frame_path)
            for obj in frame_data.get('objs', []):
                obj_type = obj.get('type')
                if obj_type in ignored_types:
                    continue
                if obj_type == 'ego-vehicle':
                    continue
                if allowed_types and obj_type not in allowed_types:
                    continue
                track_id = str(obj.get('trackID'))
                if track_id == 'None':
                    continue
                label = self._label(obj, opts)
                if label is None:
                    continue
                tracks[(drive_id, track_id)].append({
                    'frame_path': frame_path,
                    'bbox': self._xywh_to_xyxy(obj),
                    'label': label,
                    'track_id': track_id,
                })

        images = []
        bboxes = []
        pids = []
        labels = []
        track_ids = []

        for (_, track_id), entries in tracks.items():
            if len(entries) < seq_len:
                continue
            for start in range(0, len(entries) - seq_len + 1, stride):
                window = entries[start:start + seq_len]
                label = window[-1]['label']
                images.append([self._resolve_path(item['frame_path']) for item in window])
                bboxes.append([item['bbox'] for item in window])
                pids.append([[track_id] for _ in range(seq_len)])
                labels.append(label)
                track_ids.append(track_id)

        return {
            'image': images,
            'bbox': bboxes,
            'pid': pids,
            'track_id': track_ids,
            'label': labels,
        }

    def _annotation_file(self, split, opts):
        split_key = '{}_annotation_file'.format(split)
        annotation_file = opts.get(split_key)
        if annotation_file is None and split == 'val':
            annotation_file = opts.get('validation_annotation_file')
        if annotation_file is None:
            annotation_file = opts.get('annotation_file')
        if annotation_file is None:
            default_name = {'train': 'Train.json', 'val': 'Validation.json', 'test': 'Validation.json'}.get(split, 'annotations.json')
            annotation_file = os.path.join(self.data_path, default_name)
        return self._resolve_annotation_path(annotation_file)

    def _resolve_annotation_path(self, annotation_file):
        if os.path.isabs(annotation_file):
            return annotation_file
        candidates = [
            annotation_file,
            os.path.join(self.data_path, annotation_file),
            os.path.join(self.data_path, os.path.basename(annotation_file)),
        ]
        return next((p for p in candidates if os.path.exists(p)), annotation_file)

    def _is_frame_indexed(self, annotations):
        if not isinstance(annotations, dict) or not annotations:
            return False
        first_value = next(iter(annotations.values()))
        return isinstance(first_value, dict) and 'objs' in first_value

    def _records_for_split(self, annotations, split):
        if isinstance(annotations, dict):
            if split in annotations:
                return annotations[split]
            if 'tracks' in annotations:
                return [r for r in annotations['tracks'] if r.get('split') == split]
            if 'samples' in annotations:
                return [r for r in annotations['samples'] if r.get('split') == split]
        return [r for r in annotations if r.get('split') == split]

    def _frames_and_boxes(self, record):
        if 'frames' in record:
            frames = record['frames']
            if frames and isinstance(frames[0], dict):
                frame_paths = [f.get('image') or f.get('image_path') or f.get('frame_path') for f in frames]
                boxes = [f.get('bbox') or f.get('box') for f in frames]
                return frame_paths, boxes
            frame_paths = frames
        else:
            frame_paths = record.get('images') or record.get('image_paths') or record.get('frame_paths')

        boxes = record.get('bboxes') or record.get('bbox') or record.get('boxes')
        if frame_paths is None or boxes is None:
            raise ValueError('Car motion records must provide frame paths and bounding boxes.')
        return frame_paths, boxes

    def _label(self, record, opts):
        motion_map = opts.get('motion_class_map') or MOTION_TO_CLASS
        class_id_map = opts.get('class_id_map', {})
        label_source = opts.get('label_source', 'motion')

        if label_source == 'motion':
            motion = self._normalize_motion(record.get('motion'))
            if motion in motion_map:
                return int(motion_map[motion])
        if label_source == 'classID' and str(record.get('classID')) in class_id_map:
            return int(class_id_map[str(record['classID'])])

        label = record.get('motion_class', record.get('class_id', record.get('label')))
        if label is not None:
            return int(label)
        if label_source == 'classID' and record.get('classID') is not None:
            return int(record['classID'])
        if label_source == 'motion' and motion_map:
            return None
        if record.get('classID') is not None:
            return int(record['classID'])
        return None

    def _normalize_motion(self, motion):
        if motion is None:
            return None
        return str(motion).strip().lower()

    def _xywh_to_xyxy(self, obj):
        x, y, w, h = obj['xywh']
        if max(x, y, w, h) <= 1.0:
            width = obj.get('img_width', 1)
            height = obj.get('img_height', 1)
            x *= width
            w *= width
            y *= height
            h *= height
        return [x, y, x + w, y + h]

    def _drive_id(self, frame_path):
        parts = frame_path.replace('\\', '/').split('/')
        return '/'.join(parts[:-2]) if len(parts) > 2 else ''

    def _frame_sort_key(self, frame_path):
        parts = frame_path.replace('\\', '/').split('/')
        frame_name = parts[-1].split('.')[0]
        try:
            frame_idx = int(frame_name)
        except ValueError:
            frame_idx = frame_name
        return parts[:-1], frame_idx

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.data_path, path)
