import Head from 'next/head';
// import { blogPosts } from '../lib/data'
import Link from 'next/link';
import { format, parseISO, add } from 'date-fns';
import { getAllPosts } from '../lib/data';
import Image from 'next/image';
import Container from '../components/Container';

export default function Home({ posts }) {
  return (
    <Container>
      <div className="flex flex-col justify-center items-start max-w-2xl border-gray-200 dark:border-gray-700 mx-auto pb-16">
        <div className="flex flex-col-reverse sm:flex-row items-start">
          <div className="flex flex-col pr-8">
            <h1 className="font-bold text-3xl md:text-5xl tracking-tight mb-1 text-black dark:text-white">
              Sandesh Rajbhandari
            </h1>
            <h2 className="text-gray-700 dark:text-gray-200 mb-4">
              Engineer, Creator
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mb-16">
              Learning Web Dev and documenting my journey.
            </p>
          </div>
          <div className="w-[80px] sm:w-[176px] relative mb-8 sm:mb-0 mr-auto">
            <Image
              alt="Sandesh Rajbhandari"
              height={176}
              width={176}
              //public folder
              src="/avatar.jpg"
              className="rounded-full filter grayscale"
            />
          </div>
        </div>
        {posts.map((post) => (
          <BlogItem key={post.title} {...post} />
        ))}
      </div>
    </Container>
  );
}

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
        date: data.date.toISOString(),
        content,
        slug
      }))
    }
  };
}

function BlogItem({ slug, title, date }) {
  return (
    //   <div
    //     className="border border-gray-100 shadow rounded p-4
    // hover:shadow-md hover:border-gray-200 transition duration-200 ease-in"
    //   >
    //     <div>
    //       <Link href={`/blog/${slug}`}>
    //         <a className="font-bold">{title}</a>
    //       </Link>
    //     </div>
    //     <div>{format(parseISO(date), 'MMMM do, uuu')}</div>
    //   </div>
    <Link href={`/blog/${slug}`}>
      <a className="w-full">
        <div className="w-full mb-8">
          <div className="flex flex-col justify-between md:flex-row">
            <h4 className="w-full mb-2 text-lg font-medium text-gray-900 md:text-xl dark:text-gray-100">
              {title}
            </h4>
            {/* <p className="w-32 mb-4 text-left text-gray-500 md:text-right md:mb-0">
              {`${views ? new Number(views).toLocaleString() : '–––'} views`}
            </p> */}
          </div>
          <div className="text-gray-600 dark:text-gray-400">
            {format(parseISO(date), 'MMMM do, uuu')}
          </div>
          {/* <p className="text-gray-600 dark:text-gray-400">{summary}</p> */}
        </div>
      </a>
    </Link>
  );
}
