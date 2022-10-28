import ExportedImage from 'next-image-export-optimizer';

export default function BlogImg({ src, alt, layout }) {
  return (
    <div className="flex justify-center">
      <ExportedImage
        alt={alt}
        src={src}
        width="100%"
        height="100%"
        objectFit="contain"
      />
    </div>
  );
}
