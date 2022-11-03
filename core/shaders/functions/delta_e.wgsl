fn distance_cie94(one: vec3<f32>, second: vec3<f32>) -> f32{
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

    let i = pow(dL/SL, 2.0) + pow(dCab/SC, 2.0) + pow(dHab/SH, 2.0);
    if (i < 0.0) {
        return 0.0;
    } else {
        return sqrt(i);
    }
}

fn distance_cie2000(lab1: vec3<f32>, lab2: vec3<f32>) -> f32 {
    let k_L = 1.0;
    let k_C = 1.0;
    let k_H = 1.0;

    let deg360InRad = radians(360.0);
    let deg180InRad = radians(180.0);
    let pow25To7 = 6103515625.0;

	let C1 = sqrt((lab1.g * lab1.g) + (lab1.b * lab1.b));
	let C2 = sqrt((lab2.g * lab2.g) + (lab2.b * lab2.b));
    let barC = (C1 + C2) / 2.0;
    let G = 0.5 * (1.0 - sqrt(pow(barC, 7.0) / (pow(barC, 7.0) + pow25To7)));

    let a1Prime = (1.0 + G) * lab1.g;
    let a2Prime = (1.0 + G) * lab2.g;
	let CPrime1 = sqrt((a1Prime * a1Prime) + (lab1.b * lab1.b));
	let CPrime2 = sqrt((a2Prime * a2Prime) + (lab2.b * lab2.b));
    var hPrime1: f32 = 0.0;
    if (lab1.b == 0.0 && a1Prime == 0.0) {
        hPrime1 = 0.0;
    } else {
        hPrime1 = atan2(lab1.b, a1Prime);
        if (hPrime1 < 0.0) {
            hPrime1 += deg360InRad;
        }
    }
    var hPrime2: f32 = 0.0;
    if (lab2.b == 0.0 && a2Prime == 0.0) {
		hPrime2 = 0.0;
	} else {
        hPrime2 = atan2(lab2.b, a2Prime);
        if (hPrime2 < 0.0) {
            hPrime2 += deg360InRad;
        }
    }

    // Step 2
    let deltaLPrime = lab2.r - lab1.r;
    let deltaCPrime = CPrime2 - CPrime1;
    var deltahPrime = 0.0;
    let CPrimeProduct = CPrime1 * CPrime2;
    if (CPrimeProduct == 0.0) {
        deltahPrime = 0.0;
    } else {
        deltahPrime = hPrime2 - hPrime1;
        if (deltahPrime < -deg180InRad) {
            deltahPrime += deg360InRad;
        } else if (deltahPrime > deg180InRad) {
            deltahPrime -= deg360InRad;
        }
    }
    let deltaHPrime = 2.0 * sqrt(CPrimeProduct) * sin(deltahPrime/2.0);

    // Step 3
    let barLPrime = (lab1.r + lab2.r) / 2.0;
    let barCPrime = (CPrime1 + CPrime2) / 2.0;

    let hPrimeSum = hPrime1 + hPrime2;
    var barhPrime = 0.0;

    if (CPrime1 * CPrime2 == 0.0) {
        barhPrime = hPrimeSum;
    } else {
        if (abs(hPrime1 - hPrime2) <= deg180InRad) {
            barhPrime = hPrimeSum / 2.0;
        } else {
            if (hPrimeSum < deg360InRad) {
                barhPrime = (hPrimeSum + deg360InRad) / 2.0;
            } else {
                barhPrime = (hPrimeSum - deg360InRad) / 2.0;
            }
        }
    }

    let T = 1.0 - (0.17 * cos(barhPrime - radians(30.0)))
        + (0.24 * cos(2.0 * barhPrime))
        + (0.32 * cos((3.0 * barhPrime) + radians(6.0)))
        - (0.20 * cos((4.0 * barhPrime) - radians(63.0)));

    let deltaTheta = radians(30.0) * exp(-pow((barhPrime - radians(275.0))/radians(25.0), 2.0));

    let R_C = 2.0 * sqrt(pow(barCPrime, 7.0) / (pow(barCPrime, 7.0) + pow25To7));
    let S_L = 1.0 + ((0.015 * pow(barLPrime - 50.0, 2.0)) / sqrt(20.0 + pow(barLPrime - 50.0, 2.0)));
    let S_C = 1.0 + (0.045 * barCPrime);
    let S_H = 1.0 + (0.015 * barCPrime * T);
    let R_T = (-sin(2.0 * deltaTheta)) * R_C;

    let deltaE = sqrt(
	    pow(deltaLPrime / (k_L * S_L), 2.0) +
	    pow(deltaCPrime / (k_C * S_C), 2.0) +
	    pow(deltaHPrime / (k_H * S_H), 2.0) + 
	    (R_T * (deltaCPrime / (k_C * S_C)) * (deltaHPrime / (k_H * S_H))));

    return deltaE;
}