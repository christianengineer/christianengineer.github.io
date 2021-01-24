import Document, { Html, Head, Main, NextScript } from 'next/document';

class MyDocument extends Document {
  render() {
    return (
      <Html lang="en">
        <Head>
          <meta charset="UTF-8" />
          <meta
            name="viewport"
            content="width=device-width, initial-scale=1.0"
          />
          <meta http-equiv="X-UA-Compatible" content="ie=edge" />
          <title>Christian Ipanaque - Software Engineer in Seattle, WA</title>
          <link
            rel="preload"
            as="font"
            type="font/woff2"
            href="./fonts/oswald-v24-latin-regular.woff2"
            crossOrigin=""
          />

          <link
            rel="preload"
            as="font"
            type="font/woff2"
            href="./fonts/oswald-v24-latin-300.woff2"
            crossOrigin=""
          />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}

export default MyDocument;
