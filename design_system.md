## Design System: Qwen3-TTS Dashboard

### Pattern
- **Name:** Real-Time Monitoring
- **CTA Placement:** Above fold
- **Sections:** Hero > Features > CTA

### Style
- **Name:** Glassmorphism
- **Keywords:** Frosted glass, transparent, blurred background, layered, vibrant background, light source, depth, multi-layer
- **Best For:** Modern SaaS, financial dashboards, high-end corporate, lifestyle apps, modal overlays, navigation
- **Performance:** ⚠ Good | **Accessibility:** ⚠ Ensure 4.5:1

### Colors
| Role | Hex |
|------|-----|
| Primary | #3B82F6 |
| Secondary | #60A5FA |
| CTA | #F97316 |
| Background | #F8FAFC |
| Text | #1E293B |

*Notes: Dark bg + red/green alerts + trust blue*

### Typography
- **Heading:** Fira Code
- **Body:** Fira Sans
- **Mood:** dashboard, data, analytics, code, technical, precise
- **Best For:** Dashboards, analytics, data visualization, admin panels
- **Google Fonts:** https://fonts.google.com/share?selection.family=Fira+Code:wght@400;500;600;700|Fira+Sans:wght@300;400;500;600;700
- **CSS Import:**
```css
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
```

### Key Effects
Backdrop blur (10-20px), subtle border (1px solid rgba white 0.2), light reflection, Z-depth

### Avoid (Anti-patterns)
- Slow updates
- No automation

### Pre-Delivery Checklist
- [ ] No emojis as icons (use SVG: Heroicons/Lucide)
- [ ] cursor-pointer on all clickable elements
- [ ] Hover states with smooth transitions (150-300ms)
- [ ] Light mode: text contrast 4.5:1 minimum
- [ ] Focus states visible for keyboard nav
- [ ] prefers-reduced-motion respected
- [ ] Responsive: 375px, 768px, 1024px, 1440px

