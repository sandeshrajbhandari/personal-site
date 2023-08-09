import Container from '../components/Container';
import Tweet from '../components/Tweet';
// import { getTweets } from '../lib/twitter';
import Script from 'next/script'

export default function Tweets({ tweets }) {
  const tweetIds = [
      '1488096659070353413',
      '1488151330547924994',
      '1526796055890649088'
    ]
  return (
    <Container
      title="Tweets – Sandesh"
      description="A collection of tweets that inspire me, make me laugh, and make me think."
    >
      <Script
        src="https://platform.twitter.com/widgets.js"
        onLoad={() => {
          console.log('Script has loaded')
        }}
      />
      <div className="flex flex-col justify-center items-start max-w-2xl mx-auto mb-16">
        <h1 className="font-bold text-3xl md:text-5xl tracking-tight mb-4 text-black dark:text-white">
          Tweets
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          This is a collection of tweets I've enjoyed. I use Twitter quite a
          bit, so I wanted a place to publicly share what inspires me, makes me
          laugh, and makes me think.
        </p>
        {/* {tweets.map((tweet) => (
          <Tweet key={tweet.id} {...tweet} />
        ))} */}
        {tweetIds.map(id=> (
        <blockquote className="twitter-tweet text-sm">
          <a href={`https://twitter.com/x/status/${id}`}></a> 
        </blockquote>
      ))}
      </div>
    </Container>
  );
}

export async function getStaticProps() {
  // const tweets = await getTweets([
  //   '1488096659070353413',
  //   '1488151330547924994',
  //   '1526796055890649088'
  // ]);
  const tweets = "hello world"
  return { props: { tweets } };
}
