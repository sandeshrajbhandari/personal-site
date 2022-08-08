import Head from 'next/head';
// import { blogPosts } from '../lib/data'
import Link from 'next/link';
import { getAllPosts } from '../lib/data';
import ExportedImage from 'next-image-export-optimizer';

import Container from '../components/Container';
import BlogPost from '../components/BlogPost';

import { format, parseISO, add } from 'date-fns';

export default function Home({ posts }) {
  return (
    <Container>
      <div className="flex flex-col justify-around items-around customMaxWidth  mx-auto">
        {/* max-w-2xl determines the actual width of the content here. */}
        <div className="flex flex-col-reverse sm:flex-row justify-between pb-4">
          <div className="flex-1 pr-8">
            <h1 className="font-bold text-4xl text-teal-600 dark:text-white">
              Hi, I'm Sandesh. 👋
            </h1>
            <h2 className="text-gray-700 text-xl dark:text-gray-200 mb-4">
              Designer, Developer
            </h2>
            <p className="text-lg dark:text-white">
              I'm a <strong>photographer</strong> and <em>graphic designer</em>
              working with 3D art and CGI. Currently, I am learning to code
              websites and make useful web apps with functional design.
            </p>
          </div>
          <div className="w-[80px] sm:w-[176px] relative mb-8 sm:mb-0">
            <ExportedImage
              alt="Sandesh Rajbhandari"
              height={176}
              width={176}
              //public folder
              unoptimized="true"
              src="/avatar.jpg"
              className="rounded-full filter grayscale"
            />
          </div>
        </div>
        <div className="flex flex-col mx-auto w-full max-w-2xl">
          <h1 className="text-4xl font-extrabold pb-5 dark:text-white">
            Latest Articles
          </h1>
          {posts.map((post) => (
            <BlogPost key={post.title} {...post} />
          ))}
        </div>
        <div className="flex flex-col mx-auto w-full max-w-2xl">
          <h1 className="text-4xl font-extrabold pb-5 dark:text-white">
            Photography
          </h1>
          <section className="overflow-hidden text-gray-700 ">
            <div className="">
              <div className="flex flex-wrap -m-1 md:-m-2">
                <div className="flex flex-wrap w-1/2">
                  <div className="w-full p-1 md:p-2">
                    <img
                      alt="gallery"
                      className="block object-cover object-center w-full h-full rounded-lg"
                      src="https://i.imgur.com/DLvisXX.jpg"
                    />
                  </div>
                </div>
                <div className="flex flex-wrap w-1/2">
                  <div className="w-full p-1 md:p-2">
                    <img
                      alt="gallery"
                      className="block object-cover object-center w-full h-full rounded-lg"
                      src="https://i.imgur.com/QwbSY4x.jpg"
                    />
                  </div>
                </div>
                <div className="flex flex-wrap w-1/2">
                  <div className="w-full p-1 md:p-2">
                    <img
                      alt="gallery"
                      className="block object-cover object-center w-full h-full rounded-lg"
                      src="https://i.imgur.com/11Hanbs.jpg"
                    />
                  </div>
                </div>
                <div className="flex flex-wrap w-1/2">
                  <div className="w-full p-1 md:p-2">
                    <img
                      alt="gallery"
                      className="block object-cover object-center w-full h-full rounded-lg"
                      src="https://i.imgur.com/88H0TBw.jpg"
                    />
                  </div>
                </div>
              </div>
            </div>
          </section>

          <h1 className="text-4xl font-extrabold pt-4 pb-5 dark:text-white">
            3D Design
          </h1>
          <section className="overflow-hidden pb-4 text-gray-700 ">
            <div className="">
              <div className="flex flex-wrap -m-1 md:-m-2">
                <div className="flex flex-wrap w-1/3">
                  <div className="w-full p-1 md:p-2">
                    <img
                      alt="gallery"
                      className="block object-cover object-center w-full h-full rounded-lg"
                      src="https://mir-s3-cdn-cf.behance.net/project_modules/fs/a3789e126162197.6129bed3874f3.png"
                    />
                  </div>
                </div>
                <div className="flex flex-wrap w-1/3">
                  <div className="w-full p-1 md:p-2">
                    <img
                      alt="gallery"
                      className="block object-cover object-center w-full h-full rounded-lg"
                      src="https://mir-s3-cdn-cf.behance.net/project_modules/1400_opt_1/41afe9126162197.6127a8498950e.png"
                    />
                  </div>
                </div>
                <div className="flex flex-wrap w-1/3">
                  <div className="w-full p-1 md:p-2">
                    <img
                      alt="gallery"
                      className="block object-cover object-center w-full h-full rounded-lg"
                      src="https://mir-s3-cdn-cf.behance.net/project_modules/1400_opt_1/95c7a2126162197.6127a8498a637.png"
                    />
                  </div>
                </div>

                <div className="flex flex-wrap w-1/3">
                  <div className="w-full p-1 md:p-2">
                    <img
                      alt="gallery"
                      className="block object-cover object-center w-full h-full rounded-lg"
                      src="https://i.imgur.com/KuTDvUx.jpg"
                    />
                  </div>
                </div>
                <div className="flex flex-wrap w-1/3">
                  <div className="w-full p-1 md:p-2">
                    <img
                      alt="gallery"
                      className="block object-cover object-center w-full h-full rounded-lg"
                      src="https://i.imgur.com/18vJHO6.jpg"
                    />
                  </div>
                </div>
              </div>
            </div>
          </section>
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
