
# Annotation Accuracy Comparator

This script compares automated data annotation output against a manually labeled dataset and calculates:
- ✅ Overall accuracy
- 🔍 Per-label accuracy breakdown

It is useful for evaluating the performance of automated annotation tools in data labeling tasks (e.g., behavior tracking, sensor data analysis, etc.).

---

## 📦 Requirements

- Python 3.7+
- pandas (`pip install pandas`)

---

## 🚀 Usage

### Command:
```bash
python compare_annotation_accuracy_with_breakdown.py <manual_csv_file> <automated_csv_file>
```

### Example:
```bash
python compare_annotation_accuracy_with_breakdown.py P19_SJ_manual.csv P19_automated.csv
```

Make sure your CSVs have the following columns:
- `unixTimestampInMs`
- `x`, `y`, `z` (accelerometer data)
- `label`

---

## 📊 Output

The script prints:
- Total comparisons
- Total correct matches
- Overall accuracy percentage
- Per-label accuracy stats such as:
  ```
  On Task: 91.23% (1234/1353)
  Off Task: 76.45% (512/670)
  Discard: 99.32% (1444/1454)
  ```

---

## 📝 Notes

- The script matches rows using the combination of:
  `unixTimestampInMs`, `x`, `y`, and `z`.
- `label_manual` is considered ground truth; `label_automated` is the predicted label.

---

## 📄 License

© Sahil Jagtap. All rights reserved.
