import Head from 'next/head';
import Image from 'next/image';
import { getAllPosts } from '../../lib/data';
import { format, parseISO, add } from 'date-fns';
//import { data } from 'autoprefixer';
import { serialize } from 'next-mdx-remote/serialize';
import { MDXRemote } from 'next-mdx-remote';
import Container from '../../components/Container';

export default function Post({ title, date, content }) {
  return (
    <Container
      title={`${title} – Sandesh Rajbhandari`}
      description={'not mentioned'}
      //image={`https://leerob.io${post.image}`}
      image=""
      date={new Date(date).toISOString()}
      type="article"
    >
      <article className="flex flex-col items-start justify-center w-full max-w-2xl mx-auto mb-16">
        <h1 className="mb-4 text-3xl font-bold tracking-tight text-black md:text-5xl dark:text-white">
          {title}
        </h1>
        <div className="flex flex-col items-start justify-between w-full mt-2 md:flex-row md:items-center">
          <div className="flex items-center">
            <Image
              alt="Sandesh"
              height={24}
              width={24}
              src="/avatar.jpg"
              className="rounded-full"
            />
            <p className="ml-2 text-sm text-gray-700 dark:text-gray-300">
              {'Sandesh / '}
              {format(parseISO(date), 'MMMM dd, yyyy')}
            </p>
          </div>
          {/* <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 min-w-32 md:mt-0">
            {post.readingTime.text}
            {` • `}
            <ViewCounter slug={post.slug} />
          </p> */}
        </div>
        <div className="w-full mt-4 prose dark:prose-dark max-w-none">
          <MDXRemote {...content} />
        </div>
      </article>
      <div>
        <div>{format(parseISO(date), 'MMMM do, uuu')}</div>
      </div>
      {/* //here isthe hydrated content */}
    </Container>
  );
}

export async function getStaticProps(context) {
  const { params } = context; //destructure
  const allPosts = getAllPosts();
  const { data, content } = allPosts.find((item) => item.slug === params.slug);
  const mdxSource = await serialize(content);
  return {
    //props: blogPosts.find((item) => item.slug===params.slug), // will be passed to the page component as props
    props: {
      ...data,
      date: data.date.toISOString(),
      content: mdxSource
    }
  };
}
// props: blogPosts.find...... returns a blog object as prop for slug.js component.
//it destrustures it to get title, date, content and show it.

export async function getStaticPaths() {
  // const allPosts = getAllPosts();
  // console.log(allPosts);
  return {
    //   paths: [
    //     { params: { ... } }
    //   ],
    paths: getAllPosts().map((post) => ({
      //gives an array of object with params key.
      params: {
        slug: post.slug
      }
    })),
    fallback: false // false or 'blocking'
  };
  //console.log(JSON.stringify(foo,null,' '));
}
