import { format, parseISO, add } from 'date-fns';
import Link from 'next/link';

export default function BlogPost({ slug, title, date }) {
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
        <div
          className="w-full mb-8 rounded p-3 shadow-md
  "
        >
          <div className="flex flex-col justify-between md:flex-row">
            <h4 className="w-full mb-2 text-lg font-semibold text-gray-900 md:text-xl dark:text-gray-100">
              {title}
            </h4>
            {/* <p className="w-32 mb-4 text-left text-gray-500 md:text-right md:mb-0">
                {`${views ? new Number(views).toLocaleString() : '–––'} views`}
              </p> */}
          </div>
          <div className="text-gray-600 dark:text-gray-400 text-sm">
            {format(parseISO(date), 'MMMM do, uuu')}
          </div>
          {/* <p className="text-gray-600 dark:text-gray-400">{summary}</p> */}
        </div>
      </a>
    </Link>
  );
}
