import Tag from '../components/Tag';
import { getAllTags } from '../lib/tags';
import kebabCase from '../lib/utils/kebabCase';
import Container from '../components/Container';
import Link from 'next/link';

export default function Tags({ tags }) {
  const sortedTags = Object.keys(tags).sort((a, b) => tags[b] - tags[a]);
  return (
    <Container title="Tags - Sandesh Blog" description="Tags for blog posts">
      <div className="flex flex-col items-start justify-center w-full customMaxWidth mx-auto mb-16">
        <div className="self-start">
          <div className="mt-4 mb-4 text-2xl font-bold tracking-tight text-black md:text-4xl dark:text-white">
            Tags
          </div>
        </div>
        <div className="flex flex-wrap items-start">
          {Object.keys(tags).length === 0 && 'No tags found.'}
          {sortedTags.map((t) => {
            return (
              <div key={t} className="mt-2 mb-2 mr-5">
                <Tag text={t} />
                <Link
                  href={`/tags/${kebabCase(t)}`}
                  className="-ml-2 font-semibold uppercase text-gray-600 dark:text-gray-300"
                >
                  {`(${tags[t]})`}
                </Link>
              </div>
            );
          })}
        </div>
      </div>
    </Container>
  );
}

export async function getStaticProps(context) {
  const tags = await getAllTags('blog');
  return { props: { tags } };
}
