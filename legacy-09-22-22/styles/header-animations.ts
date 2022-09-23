import { css } from 'styled-components';

export const HeaderAnimations = css`
  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
  @keyframes slideInFromLeft {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(0);
    }
  }
  @keyframes slideInFromRight {
    0% {
      transform: translateX(300%);
    }
    100% {
      transform: translateX(0);
    }
  }
  @keyframes slideInFromBottom {
    0% {
      transform: rotate(-90deg) translateX(-100%);
    }
    100% {
      transform: rotate(-90deg) translateX(0);
    }
  }
  @keyframes slideInFromTop {
    0% {
      transform: rotate(-90deg) translateX(200%);
    }
    100% {
      transform: rotate(-90deg) translateX(0);
    }
  }
`;
