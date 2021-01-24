const colors = {
  primaryColor: '#3a3729',
  secondaryColor: '#fbfbfb',
  lightGrayColor: '#D8D8D8',
  accentColor: '#fff95a',
  accentSecondaryColor: '#898632',
};

const gradients = {
  light: `linear-gradient(rgba(255, 249, 90, 0.8), rgba(255, 249, 90, 0.8))`,
  dark: `linear-gradient(rgba(58, 55, 41, 0.8), rgba(58, 55, 41, 0.8))`,
};

const fonts = {
  primaryFont: `'Oswald', 'sans-serif'`,
};

const breakPoints = {
  phoneSmall: '400px',
  phoneMedium: '660px',
  phoneLarge: '760px',

  iPadMedium: '880px',
  iPadLarge: '980px',

  smallLaptop: '1080px',
  largeLaptop: '1200px',

  smallDesktop: '1400px',
  mediumDesktop: '1600px',
  largeDesktop: '1900px',
  extraLargeDesktop: '2000px',
  superExtraLargeDesktop: '2300px',
};

export default {
  breakPoints,
  colors,
  fonts,
  gradients,
};
