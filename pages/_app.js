import '../styles/global.css';
import '../styles/highlight-js/a11y-light.min.css';
import Link from 'next/link';
import { ThemeProvider } from 'next-themes';
// import { SessionProvider } from 'next-auth/react';
//import { useAnalytics } from "lib/analytics";

export default function App({ Component, pageProps }) {
  //useAnalytics(); enable later after fathom setup.
  return (
    <ThemeProvider attribute="class">
      <Component {...pageProps} />
    </ThemeProvider>
  );
}
