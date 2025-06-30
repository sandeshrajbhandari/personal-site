import React from 'react';

export default function ImageGrid({ images, columns = 3, gap = 4, maxWidth = '2xl' }) {
  // Handle different input formats
  let imageList = Array.isArray(images) ? images : [images];

  // Support both string URLs and { url, caption } objects
  imageList = imageList
    .filter(img => img && (typeof img === 'string' ? img.trim() : img.url && img.url.trim()))
    .map(img =>
      typeof img === 'string'
        ? { url: img.trim(), caption: '' }
        : { url: img.url.trim(), caption: img.caption || '' }
    );

  if (!imageList.length) {
    return null;
  }

  // Calculate minHeight for captions (e.g., 2 lines)
  const captionMinHeight = '3.5em';

  return (
    <div className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-${columns} gap-${gap} w-full max-w-${maxWidth} mx-auto my-6`}>
      {imageList.map((img, index) => (
        <div key={index} className="flex flex-col flex-1 items-center justify-start w-full">
          <div className="w-full flex items-center justify-center">
            <img
              src={img.url}
              alt={img.caption || `Image ${index + 1}`}
              className="w-full h-full object-cover rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300"
              loading="lazy"
              style={{ display: 'block' }}
            />
          </div>
          {img.caption && (
            <div
              className="mt-2 text-sm text-gray-600 dark:text-gray-300 text-center w-full"
              style={{ minHeight: captionMinHeight, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <span>{img.caption}</span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// Alternative component for more flexible grid layouts
export function FlexibleImageGrid({ images, className = '', imageClassName = '' }) {
  const imageList = Array.isArray(images) ? images : [images];
  const validImages = imageList.filter(img => img && img.trim());
  
  if (!validImages.length) {
    return null;
  }

  return (
    <div className={`grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 w-full my-6 ${className}`}>
      {validImages.map((imageUrl, index) => (
        <div key={index} className="flex justify-center">
          <img 
            src={imageUrl.trim()} 
            alt={`Image ${index + 1}`}
            className={`w-full h-auto object-cover rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300 ${imageClassName}`}
            loading="lazy"
          />
        </div>
      ))}
    </div>
  );
} 