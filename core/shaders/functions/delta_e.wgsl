fn distance_cie94(one: vec3<f32>, second: vec3<f32>) -> f32{
    let kL = 1.0;
    let K1 = 0.045;
    let K2 = 0.015;
    let dL = one.r - second.r;
    let da = one.g - second.g;
    let db = one.b - second.b;

    let C1 = sqrt(one.g * one.g + one.b * one.b);
    let C2 = sqrt(second.g * second.g + second.b * second.b);
    let dCab = C1 - C2;

    var dHab = (da * da) + (db * db) - (dCab * dCab);
    if (dHab < 0.0) {
        dHab = 0.0;
    } else {
        dHab = sqrt(dHab);
    }

    let kC = 1.0;
    let kH = 1.0;

    let SL = 1.0;
    let SC = 1.0 + K1 * C1;
    let SH = 1.0 + K2 * C1;

    let i = pow(dL/(kL * SL), 2.0) + pow(dCab/(kC * SC), 2.0) + pow(dHab/(kH * SH), 2.0);
    if (i < 0.0) {
        return 0.0;
    } else {
        return sqrt(i);
    }
}
