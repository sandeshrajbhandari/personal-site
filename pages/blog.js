import { useState } from 'react';
import { format, parseISO, add } from 'date-fns';
import Container from '../components/Container';
import BlogPost from '../components/BlogPost';
// import { InferGetStaticPropsType } from 'next';
// import { pick } from 'lib/utils';
// import { allBlogs } from '.contentlayer/data';
import { getAllPosts } from '../lib/data';

export default function Blog({ posts }) {
  const [searchValue, setSearchValue] = useState('');
  const filteredBlogPosts = posts.filter((post) =>
    post.title.toLowerCase().includes(searchValue.toLowerCase())
  );

  return (
    <Container
      title="Sandesh Blog"
      description="Thoughts on the software industry, programming, tech, videography, 3d art, and my personal life."
    >
      <div className="flex flex-col customMaxWidth w-full mx-auto">
        <h1 className="mb-4 text-3xl font-bold tracking-tight text-black md:text-5xl dark:text-white">
          🌱 Blog
        </h1>
        <p className="mb-4 text-gray-600 dark:text-gray-400">
          I started this blog to share what I learn as I document my progress as
          a web developer.<br></br> This blog is inspired by the{' '}
          <a href="https://www.swyx.io/learn-in-public/">learn in public </a>
          movement.
        </p>

        {/* search bar START */}
        <div className="relative w-full mb-4">
          <input
            aria-label="Search articles"
            type="text"
            onChange={(e) => setSearchValue(e.target.value)}
            placeholder="Search articles"
            className="block w-full px-4 py-2 text-gray-900 bg-white border border-gray-200 rounded-md dark:border-gray-900 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-800 dark:text-gray-100"
          />
          <svg
            className="absolute w-5 h-5 text-gray-400 right-3 top-3 dark:text-gray-300"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>
        {/* Search Bar END */}
        <h3 className="mt-4 mb-4 text-2xl font-bold tracking-tight text-black md:text-4xl dark:text-white">
          All Posts
        </h3>
        {!filteredBlogPosts.length && (
          <p className="mb-4 text-gray-600 dark:text-gray-400">
            No posts found.
          </p>
        )}
        {filteredBlogPosts.map((post) => (
          <BlogPost key={post.title} {...post} />
        ))}
      </div>
    </Container>
  );
}

export function getStaticProps() {
  const allPosts = getAllPosts().filter((post) => !post.data.draft);
  console.log(allPosts.map((post) => post.data.date)); //dev print statement
  const sortedPosts = allPosts.sort(
    (a, b) => Number(new Date(b.data.date)) - Number(new Date(a.data.date))
  );
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
