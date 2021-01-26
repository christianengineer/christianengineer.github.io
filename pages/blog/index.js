import { getAllPosts } from '@lib-blog';

export default function Blog({ allPosts }) {
  return (
    <>
      {allPosts.map((post) => (
        <div key={post.slug}>
          <h2>{post.title}</h2>
          <p>{post.excerpt}</p>
        </div>
      ))}
    </>
  );
}

export async function getStaticProps() {
  const allPosts = getAllPosts(['title', 'excerpt']);

  return {
    props: { allPosts },
  };
}
