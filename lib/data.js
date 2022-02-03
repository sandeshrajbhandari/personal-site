import fs from "fs";
import path from "path";
import matter from "gray-matter";

const contentDirectory = path.join(process.cwd(), "data/blog");

export function getAllPosts() {
  const allPosts = fs.readdirSync(contentDirectory);

  return allPosts.map((filename) => {
    const slug = filename.replace(".md", "");
    const fileContents = fs.readFileSync(
      path.join(contentDirectory, filename),
      "utf8"
    );
    const { data, content } = matter(fileContents);

    return {
      data,
      content,
      slug,
    };
  });
}

// export const blogPosts=[
//     {
//         title: 'Hello Post',
//         slug: 'first',
//         date: new Date().toISOString(),
//         content: '11111 lorem ipsum'
//     },
//     {
//         title: 'Second Post',
//         slug: 'second',
//         date: new Date().toISOString(),
//         content: '222222222 lorem ipsum'
//     },
//     {
//         title: 'Third Post',
//         slug: 'third',
//         date: new Date().toISOString(),
//         content: '33333333 lorem ipsum'
//     },
// ]
