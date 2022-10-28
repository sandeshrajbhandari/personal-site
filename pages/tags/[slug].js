import { useState } from 'react';
import { format, parseISO, add } from 'date-fns';
import Container from '../../components/Container';
import BlogPost from '../../components/BlogPost';
import { getAllPosts } from '../../lib/data';
import { getAllTags } from '../../lib/tags';
import kebabCase from '../../lib/utils/kebabCase';
//edited from blog.js with filtered for tag.
export default function Tag({ posts, tagName }) {
  const [searchValue, setSearchValue] = useState('');
  const filteredBlogPosts = posts.filter((post) =>
    post.title.toLowerCase().includes(searchValue.toLowerCase())
  );

  return (
    <Container title="Article Tags" description="Posts with tags custom">
      <div className="flex flex-col items-start justify-center w-full customMaxWidth mx-auto mb-16">
        <h1 className="mb-4 text-3xl font-bold tracking-tight text-black md:text-5xl dark:text-white">
          {tagName}
        </h1>
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
        {/* ------------------------------------------------------------------- */}
        <h3 className="mt-4 mb-4 text-2xl font-bold tracking-tight text-black md:text-4xl dark:text-white">
          All Posts with {tagName} tag
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

export function getStaticProps(context) {
  const { params } = context; //destructure
  const allPosts = getAllPosts();
  //const allTagPosts = allPosts.find((item) => item.slug === params.slug);
  // allPosts.map((post)=> console.log(post.data.tags))
  const allTagPosts = allPosts.filter((post) => {
    const formattedTagList = post.data.tags.map((tag) => kebabCase(tag));
    return formattedTagList.includes(params.slug);
  });
  const sortedPosts = allTagPosts.sort(
    (a, b) => Number(new Date(b.data.date)) - Number(new Date(a.data.date))
  );
  return {
    props: {
      posts: sortedPosts.map(({ data, content, slug }) => ({
        ...data,
        date: data.date,
        content,
        slug
      })),
      tagName: params.slug
    }
  };
}
// export async function getStaticProps(context) {
//     const { params } = context; //destructure
//     const allPosts = getAllPosts();
//     const { data, content } = allPosts.find((item) => item.slug === params.slug);

//     const mdxSource = await serialize(content);
//     return {
//       //props: blogPosts.find((item) => item.slug===params.slug), // will be passed to the page component as props
//       props: {
//         posts: sortedPosts.map(({ data, content, slug }) => ({
//           ...data,
//           date: data.date,
//           content,
//           slug
//         }))
//       }
//     };
//   }
// props: blogPosts.find...... returns a blog object as prop for slug.js component.
//it destrustures it to get title, date, content and show it.

export async function getStaticPaths() {
  // const allPosts = getAllPosts();
  // console.log(allPosts);
  const tagObj = await getAllTags('blog');
  const tags = Object.keys(tagObj);
  return {
    //   paths: [
    //     { params: { ... } }
    //   ],
    paths: tags.map((tagName) => ({
      //gives an array of object with params key.
      params: {
        slug: tagName
      }
    })),
    fallback: false // false or 'blocking'
  };
  //console.log(JSON.stringify(foo,null,' '));
}
