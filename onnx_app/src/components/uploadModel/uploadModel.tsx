import { Button } from '@mui/material';
import './uploadModel.css'
import { useState } from 'react';
import { LoadingButton } from '@mui/lab';

interface UploadModelParams { onUpload: (data: ArrayBuffer, classes: string[]) => void; }


const UploadModel = (params: UploadModelParams) => {
    const [model, setModel] = useState<ArrayBuffer>(new ArrayBuffer(0))
    const [classes, setClasses] = useState<string[]>(['', '', ''])

    return <div className="upload-model">
        Загрузите модель onnx:
        <br />
        <input type="file" onChange={e => {
            const file = e.target.files?.[0];
            if (file){
                const fileReader = new FileReader();
                fileReader.onload = (event) => {
                if (event.target?.result instanceof ArrayBuffer) {
                    setModel(event.target.result)
                }
                };
                fileReader.readAsArrayBuffer(file)    
            }
        }}/>

        <div className='classes-container'>
            Укажите классы для детектирования: 
            {
                classes.map((class_, ind) =>
                    <div>
                        Класс {ind}: &nbsp;
                        <input value={class_} onChange={e => {
                            setClasses(classes.map((cl, i) => ind == i ? e.target.value : cl))
                        }}/>
                    </div>
                )
            }
        </div>

        <LoadingButton 
        variant={"contained"} onClick={()=>{
            params.onUpload(model, classes);
        }}>
            Далее
        </LoadingButton>

    </div>
}

export default UploadModel