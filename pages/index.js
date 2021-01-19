const LandingPage = () => {
  return (
    <header className="header">
      <nav className="nav__top">
        <div>
          <a href="https://github.com/chrisipanaque" target="_blank">
            Github
          </a>
        </div>
        <div>
          <a href="https://www.linkedin.com/in/chrisipanaque/" target="_blank">
            LinkedIn
          </a>
        </div>
        <div>
          <a href="./christian_ipanaque_2019.pdf" target="_blank">
            Resume
          </a>
        </div>
      </nav>
      <div className="header__left">
        <h1>Christian</h1>
        <h2>Ipanaque</h2>
      </div>
      <div className="header__right"></div>
      <nav className="nav__bottom">
        <div>
          <a href="#education">Education</a>
        </div>
        <div>
          <a href="#achievements">Achievements</a>
        </div>
        <div>
          <a href="#projects">Projects</a>
        </div>
      </nav>
    </header>
  );
};

export default LandingPage;
