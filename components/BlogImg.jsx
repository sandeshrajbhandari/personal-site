import ExportedImage from 'next-image-export-optimizer';

export default function BlogImg({ src, alt, layout }) {
  // const getMeta = (url, callback) => {
  //   const img = new Image();
  //   img.src = url;
  //   img.onload = () => cb(null, img);
  // };

  // // Use like:
  // getMeta('https://i.stack.imgur.com/qCWYU.jpg', (err, img) => {
  //   console.log(img.naturalWidth, img.naturalHeight);
  // });
  return (
    <div className="flex justify-center">
      {/* <ExportedImage alt={alt} src={src} layout="fill" objectFit="contain" /> */}
      {/* produced square images with unnecessary padding. */}

      <img src={src} alt={alt} className="max-w-md" />
    </div>
  );
}
