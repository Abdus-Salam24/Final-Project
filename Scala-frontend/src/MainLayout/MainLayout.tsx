
import { Outlet } from 'react-router-dom';
import Navbar from "./Navbar";
import Footer from "./Footer";

const MainLayout: React.FC = () => {
  return (
    <div className="app-shell flex flex-col min-h-screen">
      <Navbar />
      <main className="flex-1 max-w-7xl mx-auto w-full px-3 sm:px-5 md:px-8 lg:px-10 py-5 sm:py-8 md:py-10 lg:py-12">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
};

export default MainLayout;