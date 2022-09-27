import Head from 'next/head';
import Link from 'next/link';
import { getAllPosts } from '../lib/data';
import ExportedImage from 'next-image-export-optimizer';

import Container from '../components/Container';
import BlogPost from '../components/BlogPost';

export default function Home({ posts }) {
  return (
    <Container>
      <div className="flex flex-col customMaxWidth w-full mx-auto">
        {/* max-w-2xl determines the actual width of the content here. */}
        <div className="flex flex-col-reverse sm:flex-row justify-between pt-10 pb-10">
          <div className="flex-1 pr-8">
            <h2 className="font-bold text-4xl text-teal-600 dark:text-white mb-2">
              Hi, I'm Sandesh. 👋
            </h2>
            <h2 className="text-gray-700 text-xl dark:text-gray-200 mb-4">
              Engineer, Creator
            </h2>
            <p className="text-md dark:text-white">
              I'm a mechanical engineer, developer, and creator. I write about
              web dev on my site, build projects, and dabble in 3D design and
              photography in my free time.
            </p>
            <button className="py-2 px-4 bg-blue-500">Contact Me</button>
          </div>
          <div className="w-[200px] sm:w-[200px] relative mb-8 sm:mb-0">
            <ExportedImage
              alt="Sandesh Rajbhandari"
              height={400}
              width={400}
              //public folder
              unoptimized="true"
              src="/avatar.webp"
              className="rounded-full filter grayscale"
            />
          </div>
        </div>

        {/* {------------------SECTION BREAK------------------*/}

        <div className="flex flex-col customMaxWidth mx-auto w-full max-w-2xl">
          <h1 className="text-4xl font-extrabold pb-5 dark:text-white">
            Latest Articles
          </h1>
          {posts.map((post) => (
            <BlogPost key={post.title} {...post} />
          ))}
        </div>
      </div>
    </Container>
  );
}

// export async function getStaticProps() {
//   const allPosts = getAllPosts();
//   const sortedPosts = allPosts.sort(
//     (a, b) => Number(new Date(b.data.date)) - Number(new Date(a.data.date))
//   );
//   console.log(sortedPosts);
//   //const {data, content } = allPosts.find((item) => item.slug === params.slug) ;
//   return {
//     props: {
//       posts: sortedPosts.map(({ data, content, slug }) => ({
//         ...data,
//         date: data.date.toISOString(),
//         content,
//         slug
//       }))
//     }
//   };
// }

export async function getStaticProps() {
  const allPosts = getAllPosts();
  console.log(allPosts[0].data);
  const sortedPosts = allPosts.sort(
    (a, b) => Number(new Date(b.data.date)) - Number(new Date(a.data.date))
  );
  //const {data, content } = allPosts.find((item) => item.slug === params.slug) ;
  return {
    props: {
      posts: sortedPosts.map(({ data, content, slug }) => ({
        ...data,
        date: data.date,
        content,
        slug
      }))
    }
  };
}
