import './style.css'
import { useCallback, useState } from 'react';
import axios from 'axios';

function HomePage() {

    const [text, setText] = useState('');
    const [value, setValue] = useState('');
    const [backResponse, setBackResponse] = useState(true);

    const handleNaiveBayes = useCallback(async () => {
        const backURI = "http://localhost/:5013/IA/naive-bayes/"
        try {
            const res = await axios.get(backURI + text).data;
            setValue(res)
        } catch (error) {
            setBackResponse(false);
        }
    })

    const handleDecisionTree = useCallback(async () => {
        const backURI = "http://localhost/:5013/IA/decision-tree/"
        try {
            const res = await axios.get(backURI + text).data;
            setValue(res)
        } catch (error) {
            setBackResponse(false);
        }
    })

    return (
        <div className="main">

            <textarea className='text-area' placeholder="Your Text"></textarea>
            
            <button onClick={handleNaiveBayes} onChange={(e) => setText(e.target.value)}>
                Request Naive Bayes
            </button>
            
            <button onClick={handleDecisionTree}>
                Request Decision Tree
            </button>

            <p>Value: {value}</p>
        </div>
    );
}

export default HomePage;