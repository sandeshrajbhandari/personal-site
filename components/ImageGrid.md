# ImageGrid Component Usage

The ImageGrid component allows you to display multiple images in a horizontal grid layout within your MDX blog posts.

## Basic Usage

### Simple Image Grid
```jsx
<ImageGrid 
  images={[
    'https://example.com/image1.jpg',
    'https://example.com/image2.jpg',
    'https://example.com/image3.jpg'
  ]} 
/>
```

### Custom Configuration
```jsx
<ImageGrid 
  images={['url1', 'url2', 'url3']}
  columns={2}        // Number of columns (default: 3)
  gap={6}           // Gap between images (default: 4)
  maxWidth="lg"     // Max width container (default: "md")
/>
```

### Flexible Grid (Responsive)
```jsx
<FlexibleImageGrid 
  images={['url1', 'url2', 'url3', 'url4']}
  className="max-w-4xl"           // Custom container classes
  imageClassName="border-2"       // Custom image classes
/>
```

## Features

- **Responsive Design**: Automatically adjusts columns based on screen size
- **Lazy Loading**: Images load only when needed for better performance
- **Hover Effects**: Subtle shadow effects on hover
- **Error Handling**: Gracefully handles invalid image URLs
- **Flexible Input**: Accepts single image URL or array of URLs

## Responsive Breakpoints

- Mobile: 1 column
- Small screens (sm): 2 columns  
- Large screens (lg): 3+ columns (configurable)

## Example in MDX

```mdx
# My Blog Post

Here are some images from my experiment:

<ImageGrid 
  images={[
    'https://raw.githubusercontent.com/Comfy-Org/example_workflows/main/flux/kontext/dev/rabbit.jpg',
    'https://arxiv.org/html/2506.15742v2/extracted/6566027/img/cc/img1.jpg'
  ]} 
/>

And here's a 2-column layout:

<ImageGrid 
  images={['url1', 'url2']}
  columns={2}
  gap={6}
/>
``` 