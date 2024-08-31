---
title: testing components and markdown image
date: '2022-12-29'
tags: ['next js', 'digital-garden', 'webdev']
draft: false
summary: 'testing'
---

Ever since I started learning programming and web development, I wanted a site of my own. I dabbled with blogging in the past with wordpress and other solutions, but didn't feel the sense of ownership I wanted.
This site built with code I wrote myself is my mark on the internet, that I have full control over. And here's why every aspiring developer or creator needs to have a site of their own.

# Build in Public

Learning and working on side projects in isolation can get lonely quickly, instead build in public. Do not restrict yourself to sharing the final polished product and share the W.I.Ps and issues you encounter. Who knows, maybe someone else with a similar problem in the future may benefit from your approach.

```js
function BlogImg({ src, alt, layout }) {
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
```

# Your Digital Garden

A digital garden is a space on the web where you can sow seeds for ideas that can develop over time as you grow. It is based on a popular movement with the same name that encourages people to share and promote their work on their own space. I was particularly inspired by [A Brief History & Ethos of the Digital Garden](https://maggieappleton.com/garden-history) by Maggie Appleton. Your site is your digital garden, an archive and a place to express yourself on the internet with complete control.

As for this site, here is the stack I used to build it:
[image](https://images.footlocker.com/is/image/EBFL2/C7329007)

## image using `<BlogImg />` component

<BlogImg alt="Sandesh Rajbhandari"
unoptimized="true"
src="https://www.farinasmarketing.com/wp-content/uploads/2019/07/Blogging.png"
/>

- Hosting : Netlify
- Front-end : Next.js
- Backend : Next.js with locally stored Mdx files
- Markdown to HTML : MDX-remote library
- Styling : TailwindCSS

### image using basic markdown

![testCaption](https://www.farinasmarketing.com/wp-content/uploads/2019/07/Blogging.png)
The site was based on the project structure of [leerob.io](https://leerob.io) site along with tag system implementation based on [Tailwind CSS Next.js Blog Template](https://github.com/timlrx/tailwind-nextjs-starter-blog)

![hello world](https://res.cloudinary.com/dbpb0yj1o/image/upload/v1725070089/null/qnsm6c99kjnuqxlu05dg.png)
