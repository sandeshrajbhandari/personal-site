import React from 'react';
import ImageGrid, { FlexibleImageGrid } from './ImageGrid';

// Example usage component
export default function ImageGridExample() {
  const sampleImages = [
    'https://raw.githubusercontent.com/Comfy-Org/example_workflows/main/flux/kontext/dev/rabbit.jpg',
    'https://arxiv.org/html/2506.15742v2/extracted/6566027/img/cc/img1.jpg',
    'https://via.placeholder.com/400x300/4F46E5/FFFFFF?text=Image+3',
    'https://via.placeholder.com/400x300/10B981/FFFFFF?text=Image+4'
  ];

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold mb-4">Default 3-column grid:</h3>
        <ImageGrid images={sampleImages} />
      </div>
      
      <div>
        <h3 className="text-lg font-semibold mb-4">2-column grid with custom gap:</h3>
        <ImageGrid images={sampleImages} columns={2} gap={6} />
      </div>
      
      <div>
        <h3 className="text-lg font-semibold mb-4">Flexible responsive grid:</h3>
        <FlexibleImageGrid 
          images={sampleImages} 
          className="max-w-4xl"
          imageClassName="border-2 border-gray-200"
        />
      </div>
    </div>
  );
} 