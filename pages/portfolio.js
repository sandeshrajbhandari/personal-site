import Link from 'next/link';
import Container from '../components/Container';
import ExportedImage from 'next-image-export-optimizer';

export default function About() {
  let skillList =
    'javascript, python, react, nextjs, tailwindcss, express, mongodb, Wordpress';
  // Split the string into an array of skills
  let skillsArray = skillList.split(', ');

  // Capitalize the first letter of each skill and convert the rest to lowercase
  let formattedSkills = skillsArray.map((skill) => {
    return skill.charAt(0).toUpperCase() + skill.slice(1).toLowerCase();
  });

  return (
    <Container title="Portfolio – Sandesh">
      <div className="flex flex-col justify-center items-start max-w-2xl mx-auto mb-16">
        <div>
          <h1 className="font-bold text-3xl md:text-3xl tracking-tight text-black dark:text-white">
            Sandesh Rajbhandari
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Front-end Developer
          </p>
        </div>
        <div className="prose leading-6 text-gray-600 dark:text-gray-400 mb-8">
          <p>
            Hi, I am Sandesh. I am a Front-end Developer passionate about
            building beautiful and functional websites. I am also exploring the
            world of LLMs and generative AI technologies to optimize my
            development workflow and the journey of learning new technologies.
          </p>
          <button className="btn bg-teal-600 text-white text-base">
            <Link href="/Sandesh-Dev-CV.pdf" className="text-white">
              Download CV
            </Link>
          </button>
        </div>
        <h1 className="text-3xl mb-4">My Skills</h1>
        <div id="my-skills" className="flex mx-auto gap-2">
          <div className="w-[200px] sm:w-[200px] relative flex-shrink-0 mb-8 p-2 sm:mb-0">
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
          <div className="flex flex-start flex-col justify-center flex-shrink">
            <ul className="flex flex-wrap gap-2 mb-2 space-between">
              {formattedSkills.map((skill, i) => (
                <li key={i} className="bg-teal-600 text-white px-2 py-1">
                  {skill}
                </li>
              ))}
            </ul>
            <p>
              I love building new tools with front-end technologies. Websites
              are easily shareable and accessible to people, so I love the fast
              paced development cycle of web development.
            </p>
          </div>
        </div>

        <section className="mt-8">
          <h1 className="text-3xl mb-4">My Projects</h1>
          {/* cards */}
          <div className="flex flex-col gap-4">
            <div className="bg-gray-200 p-2 rounded">
              <h3>llama-cpp-python fork</h3>
              <p>
                Fork of llama-cpp-python to enable ngrok routing to get a public
                url of the api. very basic implementation used to work with
                colab notebook to host language models while testing the api in
                my local environment.
              </p>
              <p>
                <Link href="https://github.com/sandeshrajbhandari/llama-cpp-python">
                  https://github.com/sandeshrajbhandari/llama-cpp-python
                </Link>
              </p>
            </div>
            <div className="bg-gray-200 p-2 rounded">
              <h3>yt-gpt</h3>
              <p>
                A working next.js app to summarize youtube videos using its
                transcripts. Stack used: Next.js, TailwindCSS, Next-Auth, AI SDK
                by Vercel
              </p>
              <p>
                <Link href="https://github.com/sandeshrajbhandari/yt-gpt">
                  https://github.com/sandeshrajbhandari/yt-gpt
                </Link>
              </p>
            </div>
          </div>
        </section>
      </div>
    </Container>
  );
}
