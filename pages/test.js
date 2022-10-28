import Head from 'next/head';
import Link from 'next/link';
import ExportedImage from 'next-image-export-optimizer';
import Container from '../components/Container';
export default function CardsStacked() {
  return (
    <div className="container mx-auto mt-8 flex flex-col items-center text-black p-8 border-4">
      {/* 1. Card - Stacked  */}
      <div className="max-w-sm rounded flex flex-col overflow-hidden shadow-lg">
        <h1>Card Stacked</h1>
        <img className="w-1/2 self-center" src="/team_4.jpg" />
        <div className="px-6 py-2 space-y-2 text-zinc-700">
          <h2 className="text-2xl font-sans font-bold">The Coldest Sunset</h2>
          <p className="text-gray-700 text-base">
            Lorem ipsum dolor sit amet, consectetur adipisicing
            elit.Voluptatibus quia, nulla! Maiores et
            perferendiseaque,exercitationem praesentium nihil.
          </p>
          <div className="flex justify-around text-sm text-gray-700 pt-4 pb-4">
            <span className="px-3 py-1 rounded-full bg-gray-200 font-semibold text-gray-700">
              #photography
            </span>
            <span className="px-3 py-1 rounded-full bg-gray-200 font-semibold text-gray-700">
              #travel
            </span>
            <span className="px-3 py-1 rounded-full bg-gray-200 font-semibold text-gray-700">
              #winter
            </span>
          </div>
        </div>
      </div>
      {/* 1. Card - Stacked END */}
      <div className="w-full bg-blue-500">Divider</div>
      {/* 2. Card - Horizontal */}
      <div className="max-w-sm w-full lg:max-w-full lg:flex ">
        <div
          className="h-48 lg:h-auto lg:w-48 flex-none bg-cover rounded-t lg:rounded-t-none lg:rounded-l text-center overflow-hidden bg-red-400"
          //   style={{ backgroundImage: url('/team_4.jpg') }}
          title="Woman holding a mug"
        >
          bg-image placeholder //resolve later
        </div>

        <div className="border-r border-l border-b border-gray-400 rounded-b p-4 flex flex-col justify-between leading-normal">
          <div className="mb-8">
            <p className="flex items-center text-sm text-gray-600">
              <svg
                className="fill-current text-gray-500 w-3 h-3 mr-2"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
              >
                <path d="M4 8V6a6 6 0 1 1 12 0v2h1a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2v-8c0-1.1.9-2 2-2h1zm5 6.73V17h2v-2.27a2 2 0 1 0-2 0zM7 6v2h6V6a3 3 0 0 0-6 0z" />
              </svg>
              Members Only
            </p>
            <div className="text-xl font-bold mb-2">
              Can coffee make you a better developer?
            </div>
            <p className="text-base text-gray-700">
              Lorem ipsum dolor sit amet, consectetur adipisicing elit.
              Voluptatibus quia, nulla! Maiores et perferendis eaque,
              exercitationem praesentium nihil.
            </p>
          </div>
          <div className="flex items-center">
            <img
              className="w-1- h-10 rounded-full mr-4"
              src="/team_4.jpg"
              alt="team 4 pic"
            />
            <div className="flex flex-col justify-start">
              <span className="leading-none">Jonathan Reinink</span>
              <span className="text-gray-400">Aug 18</span>
            </div>
          </div>
        </div>
      </div>
    </div> // main page container div
  );
}
