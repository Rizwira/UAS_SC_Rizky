const tf = require('@tensorflow/tfjs-node');

function normalized(data){ // x & y
    x1 = (data[0] - 42.66333) / 10.61528
    x2 = (data[1] - 88.73667) / 19.01403 
    x3 = (data[2] - 143.0478) / 23.06124 
    return [x1, x2, x3]
}

function denormalized(data){
    y1 = (data[0] * 9.184435) + 74.73778
    y2 = (data[1] * 14.72226) + 49.82889
    y3 = (data[2] * 23.97945) + 159.7089
    return [y1, y2, y3]
}


async function predict(data){
    let in_dim = 3;
    
    data = normalized(data);
    shape = [1, in_dim];

    tf_data = tf.tensor2d(data, shape);

    try{
        // path load in public access => github
        const path = 'https://raw.githubusercontent.com/Rizwira/UAS_SC_Rizky/main/public/ex_model/model.json';
        const model = await tf.loadGraphModel(path);
        
        predict = model.predict(
                tf_data
        );
        result = predict.dataSync();
        return denormalized( result );
        
    }catch(e){
      console.log(e);
    }
}

module.exports = {
    predict: predict 
}
