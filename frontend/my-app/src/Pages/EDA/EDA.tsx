import { TIMEOUT } from "dns";

/**
 * @component
 * @returns {React.ReactElement} A page with full responsive css
 */
export default function EDA() {
    return (
        <div style={{ width: '100%', height: '100vh' }}>
        <h1>Đây là phần EDA</h1>
        <iframe
          src="/notebook/EDA.html"
          title="EDA Notebook"
          style={{ width: '100%', height: '100%', border: 'none' }}
        />
      </div>
    );
}
