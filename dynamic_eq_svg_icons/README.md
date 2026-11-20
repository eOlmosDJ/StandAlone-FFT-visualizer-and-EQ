# Dynamic EQ — SVG icon collection

- 25 individual SVG files, each with a 48×48 viewBox.
- Minimal line-art style with multicolor semantic accents.
- Transparent backgrounds; paths and strokes remain editable.
- `preview_contact_sheet.svg` is a visual overview.

## Integration
Use each SVG as a `QIcon` in PyQt/PySide, or render it with `QSvgRenderer`.
For a dropdown control, put the icon on a `QToolButton` and attach a `QMenu`.
The filter icons are intentionally schematic response curves, not calibrated plots.

## Suggested menu mappings
- Stereo mode: stereo, left, right, mid, side
- A/B mode: ab, a, b
- Filter type: bell, low_shelf, high_shelf, low_cut, high_cut, notch, band_pass
- Power state: on, off
