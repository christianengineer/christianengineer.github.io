// var element = document.querySelector(".top-hire-now");

const animation = anime({
  targets: ".profile-image",
  translateX: [-200, 0],
  delay: function (el, i) {
    return i * 100;
  },
  elasticity: 200,
  easing: "easeInSine",
  autoplay: false,
});

window.onscroll = function (e) {
  console.log("what");
  animation.seek(window.scrollY * 1.85);
};
