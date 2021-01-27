import { getAllPosts } from '@blog-library';
import { AllPosts } from '@blog-components';

export default function Blog({ allPosts }) {
  return (
    <AllPosts>
      {allPosts.map((post) => (
        <AllPosts.Post key={post.slug}>
          <h2>{post.title}</h2>
          <p>{post.excerpt}</p>
        </AllPosts.Post>
      ))}
    </AllPosts>
  );
}

export async function getStaticProps() {
  const allPosts = getAllPosts(['title', 'slug', 'excerpt']);

  return {
    props: { allPosts },
  };
}
