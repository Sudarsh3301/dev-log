# Design System Inspired by Arya's Portfolio

## 1. Visual Theme & Atmosphere

This design system embodies minimalist sophistication with a technical, approachable aesthetic. The interface prioritizes clarity and readability through a monospaced typeface, creating a developer-friendly atmosphere that reflects the portfolio owner's Computer Engineering background. The color palette is deliberately restrained—dominated by neutral grays with subtle lavender accents—allowing content to breathe while maintaining visual hierarchy. The overall mood is calm, focused, and intellectually engaged, with an emphasis on whitespace and deliberate typography rather than decorative elements. This is a space for serious work and thoughtful communication, where every design decision serves a functional purpose.

**Key Characteristics**
- Monospaced typography throughout (JetBrains Mono)
- High contrast neutral base with selective lavender accents
- Generous whitespace and breathing room
- Minimal decoration; form follows function
- Tech-forward, academic aesthetic
- Clean horizontal navigation with restrained styling
- Circular profile imagery (50% border radius)
- Soft gray tones suggesting approachability despite technical focus

## 2. Color Palette & Roles

### Primary
- **Accent Lavender** (`#A898E9`): Primary interactive accent color used for links, highlights, and emphasis elements. Signals interactivity and draws attention to key content areas.

### Neutral Scale
- **Dark Gray** (`#666666`): Primary text color for body content and standard UI elements. High contrast against light backgrounds for readability.
- **Medium Gray** (`#807E7E`): Secondary text, meta information, and reduced-emphasis content. Used extensively for supporting text and icons.
- **Off-White** (`#FEFEFE`): Subtle background wash, near-white with minimal warmth.
- **Light Gray** (`#EEEEEE`): Secondary background, borders, and divider lines.
- **Pure White** (`#FFFFFF`): Primary background, cards, and containers.

### Surface & Borders
- **Border Gray** (`#EEEEEE`): Dividers and subtle borders between sections.

## 3. Typography Rules

### Font Family
- **Primary Font**: JetBrains Mono, monospace
- **Fallback Stack**: `'JetBrains Mono', 'Courier New', monospace`

### Hierarchy

| Role | Font | Size | Weight | Line Height | Letter Spacing | Notes |
|------|------|------|--------|-------------|-----------------|-------|
| Display / H1 | JetBrains Mono | 17px | 700 | 1 (normal) | 0px | Page title and major headings; compact vertical spacing |
| Body | JetBrains Mono | 16px | 400 | 24px | 0px | Primary content text; 1.5x line-height for readability |
| Link | JetBrains Mono | 16px | 400 | normal | 0px | Navigation and inline links; inherits body baseline |
| Caption | JetBrains Mono | 14px | 400 | 20px | 0px | Supporting text, metadata, and annotations |
| Code | JetBrains Mono | 13px | 400 | 18px | 0px | Inline code and technical references |

### Principles
- All typography uses monospaced JetBrains Mono for technical authenticity and consistency
- Hierarchy is achieved through weight (400 vs 700) and size rather than font changes
- Line heights are generous in body text (1.5x) for comfortable reading despite monospace font
- Headings use minimal line height (1.0) for compact, punchy presentation
- Letter spacing remains at 0px throughout—monospace naturally provides character separation

## 4. Component Stylings

### Buttons
Currently not present in primary design. If needed, would follow link styling with added padding and border radius for tactile feedback.

### Links & Navigation
- **Default State**
  - Background: `transparent`
  - Text Color: `#666666`
  - Font: JetBrains Mono, 16px, weight 400
  - Padding: `0px 0px 0px 0px`
  - Border: `none`
  - Border Radius: `0px`
  - Box Shadow: `none`
  - Line Height: `normal`
  - Decoration: Underline on hover (inferred)

- **Hover State** (inferred)
  - Text Color: `#A898E9`
  - Transition: Smooth color change (recommended 200ms)

### Profile Section
- **Avatar**
  - Border Radius: `50%`
  - Width & Height: Inferred square `80px`
  - Border: `2px solid #EEEEEE`
  - Box Shadow: `none`
  - Background: Image-based

### Text Containers
- **Default**
  - Background: `#FFFFFF`
  - Padding: `32px`
  - Border: `none`
  - Border Radius: `0px`
  - Text Color: `#666666`
  - Font Size: `16px`
  - Line Height: `24px`

### Headings (H2, H3)
- **Default**
  - Background: `transparent`
  - Text Color: `#666666`
  - Font: JetBrains Mono, 17px, weight 700
  - Padding: `0px`
  - Line Height: `normal`
  - Margin Bottom: `16px` (inferred for spacing)

### Accent Links (world model, featured keywords)
- **State: Default**
  - Text Color: `#A898E9`
  - Background: `transparent`
  - Font Weight: `400`
  - Text Decoration: `none`
  - Border: `none`

- **State: Hover**
  - Text Color: `#A898E9` (sustained)
  - Text Decoration: `underline` (inferred)
  - Opacity: Possible slight increase

## 5. Layout Principles

### Spacing System
- **Base Unit**: 16px
- **Scale**: Multiples observed are 32px (2x) and 64px (4x)
- **Usage Context**:
  - 32px: Primary internal padding, section padding
  - 64px: Page-level margins, section separation

### Grid & Container
- **Max Width**: Likely 1000-1200px (inferred from centered content layout)
- **Column Strategy**: Single centered column with symmetric margins
- **Section Patterns**: Full-width sections with 32px internal padding, 64px vertical separation between major sections

### Whitespace Philosophy
Whitespace is treated as a first-class design element. The layout embraces negative space to reduce cognitive load and emphasize content hierarchy. Generous margins around content blocks create visual breathing room, particularly in the navigation area and between biographical sections. This approach reflects the minimalist design ethos.

### Border Radius Scale
- **Image Elements**: `50%` (perfect circles for avatars)
- **Standard Components**: `0px` (sharp corners throughout, no border radius on text elements or containers)

## 6. Depth & Elevation

| Level | Treatment | Use |
|-------|-----------|-----|
| Flat (0) | No shadow, `box-shadow: none` | Primary text, links, backgrounds; flat design approach |
| Subtle (1) | `box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.08)` | Card hover states or subtle lift (if needed) |
| Raised (2) | `box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.12)` | Modal or prominent overlay (if implemented) |

**Shadow Philosophy**: This design system maintains a flat aesthetic with no elevation shadows in the base design. The interface relies on color, typography, and whitespace for hierarchy rather than depth cues. If shadows become necessary for interactive states or modal dialogs, they should be subtle and restrained to maintain the minimalist visual integrity.

## 7. Do's and Don'ts

### Do
- Use JetBrains Mono for all typography—maintain monospace consistency
- Apply 32px padding within content sections
- Maintain high contrast between text and background for accessibility
- Use `#A898E9` accent color sparingly for key interactions and links
- Embrace whitespace; don't crowd content
- Keep border radius at 0px for containers and text elements; use 50% only for avatars
- Use weight 700 for headings, weight 400 for body and links
- Maintain 24px line height for body text for readability
- Keep shadows minimal or absent; rely on color and spacing for hierarchy

### Don't
- Mix fonts or use sans-serif alternatives
- Add unnecessary decorative elements or borders
- Use drop shadows as primary depth indicator
- Apply strong color saturation or vivid accent colors beyond `#A898E9`
- Create narrow line heights in body text (minimum 1.5x)
- Use rounded corners on text containers or UI elements
- Apply background colors to links except on hover/active states
- Introduce opacity changes on neutral text elements
- Ignore the monospace baseline; maintain consistent character alignment

## 8. Responsive Behavior

### Breakpoints

| Breakpoint Name | Width | Key Changes |
|-----------------|-------|------------|
| Mobile | 320px–640px | Single column, reduced padding to 24px, font sizes scale by 85%, navigation stacks vertically |
| Tablet | 641px–1024px | Moderate padding 28px, max-width adjusted to 90% viewport, line heights maintained |
| Desktop | 1025px+ | Full 32px padding, max-width 1000px–1200px centered, navigation horizontal |

### Touch Targets
- **Minimum Touch Target Size**: 44px × 44px for interactive elements (links, buttons)
- **Spacing Between Targets**: Minimum 8px to prevent accidental activation
- **Avatar Clickable Area**: Ensure 80px minimum diameter for touch usability

### Collapsing Strategy
- Stack all navigation links vertically on mobile (≤640px) with 16px vertical spacing
- Reduce heading font size to 15px on mobile for screen fit
- Decrease body padding to 24px on mobile while maintaining 24px line height
- Profile section remains centered but may reduce avatar to 64px on small phones
- Maintain 64px vertical section separation on mobile for visual rhythm

## 9. Agent Prompt Guide

### Quick Color Reference
- **Primary CTA / Accent**: Accent Lavender (`#A898E9`) — for links, highlights, and emphasis
- **Body Text**: Dark Gray (`#666666`) — primary readable content
- **Secondary Text**: Medium Gray (`#807E7E`) — meta information and reduced emphasis
- **Background**: Pure White (`#FFFFFF`) — primary surface
- **Subtle Background**: Off-White (`#FEFEFE`) — secondary surface or section wash
- **Borders & Dividers**: Light Gray (`#EEEEEE`) — subtle separation lines

### Iteration Guide
1. **Use JetBrains Mono exclusively** — all typography must be monospaced, weight 400 (regular) or 700 (bold)
2. **Headings are 17px, weight 700, line-height normal** — compact and prominent
3. **Body text is 16px, weight 400, line-height 24px** — prioritize readability over density
4. **Links are 16px, color `#666666`, no underline by default** — become `#A898E9` on hover
5. **Padding scale: 32px for sections, 64px for vertical separation** — use these consistently
6. **No border radius except avatars at 50%** — maintain flat, sharp aesthetic
7. **No box shadows in base design** — rely on color, typography, and whitespace for hierarchy
8. **Maintain 1:1 aspect ratios for avatar circles** — perfect geometric balance
9. **Center all content within max-width container** — symmetric margins on desktop
10. **Preserve ample whitespace** — don't reduce padding below 32px on desktop or below 24px on mobile