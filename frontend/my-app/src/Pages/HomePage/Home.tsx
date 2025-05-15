import { TIMEOUT } from "dns";
import './Home.css'
/**
 * @component
 * @returns {React.ReactElement} A home page with full responsive css
 */
export default function Home() {
    return (
        <div className="home-page">
            <h1>CS114 Project</h1>
            <h1>Dataset: Framingham heart study Dataset</h1>
            <h1>Heart Attack Prediction</h1>
            <h1>Thành viên</h1>
            <h3> Đỗ Phương Duy - 23520362 </h3>
            <h3> Bùi Ngọc Thiên Thanh - 23521436 </h3> 
            <h3> Đặng Quang Vinh - 23521786 </h3>
        </div>
    );
}
