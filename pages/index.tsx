import React from "react";
import { getAllPosts } from "../lib/api";
import Post from "../interfaces/post";

type IndexProps = {
  allPosts: Post[];
};

export default function Index({ allPosts }: IndexProps) {
  const heroPost = allPosts[0];
  return <div>{heroPost.title}</div>;
}

export const getStaticProps = async () => {
  const allPosts = getAllPosts(["title"]);

  return {
    props: {
      allPosts,
    },
  };
};
