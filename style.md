# QuickFitter Style Guide

This document outlines the design system and styling patterns used in the QuickFitter application. Follow these guidelines to maintain visual consistency across all pages.

## Core Design Principles

- **Clean and Professional**: Minimalist design with subtle shadows and borders
- **Responsive**: Mobile-first approach with appropriate breakpoints
- **Accessible**: High contrast ratios and clear visual hierarchy
- **Interactive**: Smooth transitions and clear hover states

## Technology Stack

- **CSS Framework**: Tailwind CSS (latest version via CDN - `https://cdn.tailwindcss.com`)
- **Font**: Inter (Google Fonts)
- **Icons**: Inline SVG for consistency
- **Charts**: Plotly.js for data visualization

> **Note**: The project uses Tailwind CSS from the CDN without version pinning, which means it uses the latest stable version. This ensures access to the newest features but may require updates if breaking changes occur.

## Color Palette

### Primary Colors
- **Blue-600**: `#2563eb` - Primary actions (buttons, links)
- **Blue-700**: `#1d4ed8` - Hover states
- **Blue-50**: `#eff6ff` - Light backgrounds
- **Blue-200**: `#bfdbfe` - Borders

### Neutral Colors
- **Gray-100**: `#f3f4f6` - Page background
- **Gray-200**: `#e5e7eb` - Borders, dividers
- **Gray-300**: `#d1d5db` - Subtle borders
- **Gray-400**: `#9ca3af` - Muted text
- **Gray-500**: `#6b7280` - Secondary text
- **Gray-600**: `#4b5563` - Body text
- **Gray-700**: `#374151` - Headings
- **Gray-800**: `#1f2937` - Primary text
- **Gray-900**: `#111827` - Dark backgrounds

### Status Colors
- **Green-500**: `#10b981` - Success/Excellent
- **Yellow-500**: `#f59e0b` - Warning/Good
- **Red-500**: `#ef4444` - Error/Poor
- **Green-50/200**: Light success backgrounds/borders
- **Yellow-50/200**: Light warning backgrounds/borders
- **Red-50/200**: Light error backgrounds/borders

## Typography

### Font Setup
```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
```

### Font Sizes
- **Headings**:
  - H1: `text-3xl md:text-4xl font-bold` (30px/36px on mobile, 36px/40px on desktop)
  - H2: `text-2xl font-bold` (24px)
  - H3: `text-xl font-bold` (20px)
  - H4: `text-lg font-semibold` (18px)
  - H5: `text-sm font-semibold` (14px)

- **Body Text**:
  - Default: `text-sm` (14px)
  - Small: `text-xs` (12px)
  - Large: `text-lg` (18px)

### Text Colors
- Primary text: `text-gray-800`
- Secondary text: `text-gray-600`
- Muted text: `text-gray-500`
- Dark backgrounds: `text-white`

## Components

### Buttons

#### Primary Button
```html
<button class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg shadow-md transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50">
  Action
</button>
```

#### Secondary Button
```html
<button class="bg-gray-200 hover:bg-gray-300 text-gray-700 px-3 py-1 rounded text-sm transition-colors">
  Secondary Action
</button>
```

#### Small Button
```html
<button class="bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 text-sm rounded">
  Apply Changes
</button>
```

### Cards

#### Basic Card
```html
<div class="bg-white rounded-xl shadow-lg p-3 md:p-4">
  <!-- Content -->
</div>
```

#### Floating Card
```html
<div class="fixed rounded-lg bg-white shadow-lg overflow-hidden">
  <!-- Header -->
  <div class="flex items-center justify-between bg-gray-100 border-b px-3 py-2">
    <h4 class="font-semibold text-sm text-gray-700">Title</h4>
  </div>
  <!-- Content -->
  <div class="p-3">
    <!-- ... -->
  </div>
</div>
```

### Form Elements

#### Input Fields
```html
<input type="text" class="block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 text-sm">
```

#### Select Dropdowns
```html
<select class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md">
  <option>Option</option>
</select>
```

#### File Input
```html
<input type="file" class="block w-full text-sm text-gray-500
  file:mr-4 file:py-2 file:px-4
  file:rounded-lg file:border-0
  file:text-sm file:font-semibold
  file:bg-blue-50 file:text-blue-700
  hover:file:bg-blue-100">
```

#### Range Sliders
```html
<input type="range" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
```

### Labels
```html
<label class="block text-sm font-medium text-gray-700 mb-2">
  Label Text
</label>
```

### Status Indicators

#### Success
```html
<div class="p-3 rounded-lg bg-green-50 border border-green-200">
  <span class="text-green-700">Success message</span>
</div>
```

#### Warning
```html
<div class="p-3 rounded-lg bg-yellow-50 border border-yellow-200">
  <span class="text-yellow-700">Warning message</span>
</div>
```

#### Error
```html
<div class="p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
  <h3 class="font-bold">Error</h3>
  <p>Error message</p>
</div>
```

### Progress Bars
```html
<div class="w-full bg-gray-200 rounded-full h-2">
  <div class="h-2 rounded-full transition-all duration-300 bg-blue-500" style="width: 75%"></div>
</div>
```

## Layout Patterns

### Container
```html
<div class="container mx-auto p-4 md:p-6 lg:p-8 max-w-7xl">
  <!-- Content -->
</div>
```

### Split Pane
```html
<div class="flex h-full w-full overflow-hidden">
  <div class="flex-1 overflow-auto p-4">
    <!-- Left content -->
  </div>
  <div class="w-1 bg-gray-300 cursor-col-resize hover:bg-gray-400"></div>
  <div class="flex-1 overflow-auto p-4">
    <!-- Right content -->
  </div>
</div>
```

### Floating Elements
```html
<div class="fixed top-4 left-4 z-50">
  <!-- Floating content -->
</div>
```

## Interactive Elements

### Hover States
- Buttons: Darker shade with `transition-colors duration-300`
- Cards: Shadow increase with `hover:shadow-xl`
- Links: `hover:underline` or color change

### Focus States
- Form elements: Blue ring with `focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50`
- Buttons: Same as form elements
- Remove default outline: `focus:outline-none`

### Transitions
- Color changes: `transition-colors duration-300`
- Shadow changes: `transition-shadow duration-200`
- Transform changes: `transition-transform duration-200`
- All properties: `transition-all duration-300`

## Spacing System

### Padding
- Small: `p-2` (8px)
- Medium: `p-3` (12px) or `p-4` (16px)
- Large: `p-6` (24px) or `p-8` (32px)
- Responsive: `p-2 md:p-4` (8px mobile, 16px desktop)

### Margins
- Between sections: `mt-8` (32px)
- Between elements: `mt-4` (16px) or `space-y-4`
- Small gaps: `mt-2` (8px) or `space-y-2`

### Gaps in Flex/Grid
- `gap-4` (16px) or `gap-6` (24px)
- `space-x-2` (8px horizontal)
- `space-y-4` (16px vertical)

## Borders and Shadows

### Borders
- Default: `border border-gray-200`
- Subtle: `border border-gray-300`
- Dividers: `border-t` or `border-b`

### Border Radius
- Small: `rounded` (4px)
- Medium: `rounded-lg` (8px)
- Large: `rounded-xl` (12px)
- Full: `rounded-full` (circles)

### Shadows
- Small: `shadow` or `shadow-sm`
- Medium: `shadow-md`
- Large: `shadow-lg`
- Extra large: `shadow-xl`
- Hover effect: `hover:shadow-xl`

## Icons

### SVG Pattern
```html
<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="..."/>
</svg>
```

### Common Sizes
- Small: `w-4 h-4` (16px)
- Medium: `w-5 h-5` (20px)
- Large: `w-6 h-6` (24px)

## Responsive Design

### Breakpoints
- Mobile: Default styles
- Tablet: `md:` prefix (768px+)
- Desktop: `lg:` prefix (1024px+)

### Common Patterns
```html
<!-- Text size responsive -->
<h1 class="text-2xl md:text-3xl lg:text-4xl">

<!-- Padding responsive -->
<div class="p-2 md:p-4 lg:p-6">

<!-- Grid responsive -->
<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
```

## Animation

### Loading Spinner
```css
.spinner {
  border-top-color: #3498db;
  animation: spin 1s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
```

### Smooth Scrolling
```css
html { scroll-behavior: smooth; }
```

## Best Practices

1. **Consistency**: Use the predefined color palette and spacing system
2. **Accessibility**: Maintain high contrast ratios and provide focus indicators
3. **Performance**: Use Tailwind's utility classes instead of custom CSS when possible
4. **Responsiveness**: Always consider mobile-first design
5. **Interactivity**: Provide clear hover and focus states
6. **Hierarchy**: Use consistent heading sizes and font weights
7. **Whitespace**: Use generous spacing for better readability
8. **Feedback**: Provide visual feedback for all interactive elements

## Implementation Notes

1. Include Tailwind CSS via CDN:
```html
<script src="https://cdn.tailwindcss.com"></script>
```

2. Set the base font and background:
```html
<body class="bg-gray-100 text-gray-800">
```

3. Use semantic HTML elements with Tailwind utility classes
4. Avoid inline styles except for dynamic values
5. Group related utility classes logically
6. Use responsive prefixes consistently

This style guide ensures visual consistency and professional appearance across all pages of the application.