# VectorTrade - Design System Specifications (Apple HIG + Linear Precision)

## 1. Visual Theme & Polarities
- **Background Canvas**: Deep Void Black `#000000` (Apple Pro display pure black).
- **Surface Elevation 1**: `rgba(28, 28, 30, 0.65)` with backdrop-filter: blur(24px).
- **Surface Elevation 2**: `rgba(44, 44, 46, 0.45)`.
- **Dividers & Borders**: 1px Hairline `rgba(255, 255, 255, 0.08)`.
- **Interactive Accent**: Action Blue `#0A84FF` / Indigo `#5E6AD2`.
- **Status Green**: Apple SF Emerald `#30D158`.
- **Status Amber**: Apple Warning Amber `#FF9F0A`.
- **Status Coral/Red**: Apple Error Red `#FF453A`.

## 2. Typography
- **Font Stack**: `-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif`.
- **Monospace Stack**: `"SF Mono", Menlo, Monaco, Consolas, "Courier New", monospace`.
- **Letter Spacing**: `-0.022em` on titles, `+0.01em` on uppercase micro-badges.

## 3. Motion & Springs (Apple Motion Skill)
- **Fluid Spring**: `cubic-bezier(0.16, 1, 0.3, 1)`
- **Duration**: `240ms` for interactions, `380ms` for panel disclosures.
- **Zero Jitter**: Direct manipulation, no layout shift or flashing DOM rebuilds.
- **No-Slop Standard**: Clean restrained typography, honest real metrics, crisp data tables.
