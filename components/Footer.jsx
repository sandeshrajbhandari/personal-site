import Link from 'next/link';

const ExternalLink = ({ href, children }) => (
  <div
    className="text-gray-500 hover:text-gray-600 transition"
    target="_blank"
    rel="noopener noreferrer"
    href={href}
  >
    {children}
  </div>
);

export default function Footer() {
  return (
    <footer className="flex flex-col justify-center items-center customMaxWidth mx-auto w-full mb-8">
      <hr className="w-full border-1 border-gray-200 dark:border-gray-800 mb-8" />

      <div className="w-full max-w-2xl grid grid-cols-1 gap-4 pb-16 sm:grid-cols-3">
        <div className="flex flex-col space-y-4 items-center">
          <Link href="/">
            <div className="text-gray-500 hover:text-gray-600 transition">Home</div>
          </Link>
          <Link href="/about">
            <div className="text-gray-500 hover:text-gray-600 transition">
              About
            </div>
          </Link>
        </div>
        {/* next-column */}
        <div className="flex flex-col space-y-4 items-center">
          <ExternalLink href="https://twitter.com/sandeshrajx">
            Twitter
          </ExternalLink>
          <ExternalLink href="https://github.com/sandeshrajbhandari">
            GitHub
          </ExternalLink>
          {/* Youtube Link add */}
        </div>
        <div className="flex flex-col space-y-4 items-center">
          <Link href="/tweets">
            <div className="text-gray-500 hover:text-gray-600 transition">
              Tweets(to-do)
            </div>
          </Link>
          <Link href="/stack">
            <div className="text-gray-500 hover:text-gray-600 transition">
              My Stack
            </div>
          </Link>
        </div>
      </div>
    </footer>
  );
}
