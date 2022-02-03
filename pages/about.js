import Link from "next/link";
import Container from "components/Container";

export default function About() {
  return (
    <Container title="About – Sandesh">
      <div className="flex flex-col justify-center items-start max-w-2xl mx-auto mb-16">
        <h1 className="font-bold text-3xl md:text-5xl tracking-tight mb-4 text-black dark:text-white">
          About Me
        </h1>
        <div className="mb-8 prose leading-6 text-gray-600 dark:text-gray-400">
          <p>
            Hi, I am Sandesh, a Mechanical Engineer by profession, and
            passionate about programming and design.
          </p>
          <p>
            I started learning to code since high school, but never had the time
            or patience to dedicate my self to the craft. This site is my
            attempt at getting back in to the world of coding and track my
            progress.
          </p>
          <p>
            I have a Mechanical Engineering Degree from Institute of
            Engineering, Tribhuwan University. I spend my free time reading
            books, messing with 3D Art, Photography and coding.
          </p>
        </div>
        {/* <iframe
          height="280"
          src="https://www.google.com/maps/d/embed?mid=1QOGi-u8d4vwoQ4vC4zQjKxrSfsDIQdOK&hl=en"
          title="Lee's Travel Map"
          width="100%"
        /> */}
      </div>
    </Container>
  );
}
