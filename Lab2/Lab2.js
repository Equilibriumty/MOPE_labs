// constants
let variant = 11
let m = 5
let maxY = (30 - variant) * 10
let minY = (20 - variant) * 10
let x1Min = 10
let x1Max = 60
let x2Min = -30
let x2Max = 45
let normalize_x = [[-1, -1], [1, 1], [-1, 1]]

const getRandomInt = (min, max) => {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

const getAverageY = arr => {
    let averageY = []
    for (let i = 0; i < arr.length; i++) {
        let s = 0
        for (j in arr[i]) {
            s += j
        }
        averageY.push(s / (arr[i].length))
    } return averageY
}

const getDispersion = arr => {
    let dispersion = []
    for (let i = 0; i < arr.length; i++) {
        let s = 0
        for (j in arr[i]) {
            s += (j - getAverageY(arr)[i]) * (j - getAverageY(arr)[i])
        }
        dispersion.push(s / (arr[i].length))
    } return dispersion
}

const calculateFuv = (u, v) => {
    if (u >= v) {
        return u / v
    } else return v / u
}
const getDiscriminant = (x11, x12, x13, x21, x22, x23, x31, x32, x33) => x11 * x22 * x33 + x12 * x23 * x31 + x32 * x21 * x13 - x13 * x22 * x31 - x32 * x23 * x11 - x12 * x21 * x33

let Y = []
for (let i = 0; i < 4; i++) {
    Y[i] = []
    for (let j = 0; j < m; j++) {
        Y[i][j] = getRandomInt(minY, maxY)
    }
}
let avrY = getAverageY(Y)
let sigmaOfTeta = Math.sqrt((2 * (2 * m - 2)) / (m * (m - 4)))
let F_uv = []
let teta = []
let R_uv = []
// F_uv
F_uv.push(calculateFuv(getDispersion(Y)[0], getDispersion(Y)[1]))
F_uv.push(calculateFuv(getDispersion(Y)[2], getDispersion(Y)[0]))
F_uv.push(calculateFuv(getDispersion(Y)[2], getDispersion(Y)[1]))
// teta
teta.push(((m - 2) / m) * F_uv[0])
teta.push(((m - 2) / m) * F_uv[1])
teta.push(((m - 2) / m) * F_uv[2])
// R_uv
R_uv.push(Math.abs(teta[0] - 1) / sigmaOfTeta)
R_uv.push(Math.abs(teta[1] - 1) / sigmaOfTeta)
R_uv.push(Math.abs(teta[2] - 1) / sigmaOfTeta)
let Rkr = 2
for (let i = 0; i < R_uv.length; i++) {
    if (R_uv[i] > Rkr) {
        console.error("Repeat experiment | Error")
    }
}
let mx1 = (normalize_x[0][0] + normalize_x[1][0] + normalize_x[2][0]) / 3
let mx2 = (normalize_x[0][1] + normalize_x[1][1] + normalize_x[2][1]) / 3
let my = (avrY[0] + avrY[1] + avrY[2]) / 3

let a1 = (normalize_x[0][0] ** 2 + normalize_x[1][0] ** 2 + normalize_x[2][0] ** 2) / 3
let a2 = (normalize_x[0][0] * normalize_x[0][1] + normalize_x[1][0] * normalize_x[1][1] + normalize_x[2][0] * normalize_x[2][1]) / 3
let a3 = (normalize_x[0][1] ** 2 + normalize_x[1][1] ** 2 + normalize_x[2][1] ** 2) / 3

let a11 = (normalize_x[0][0] * avrY[0] + normalize_x[1][0] * avrY[1] + normalize_x[2][0] * avrY[2]) / 3
let a22 = (normalize_x[0][1] * avrY[0] + normalize_x[1][1] * avrY[1] + normalize_x[2][1] * avrY[2]) / 3

let b0 = getDiscriminant(my, mx1, mx2, a11, a1, a2, a22, a2, a3) / getDiscriminant(1, mx1, mx2, mx1, a1, a2, mx2, a2, a3)
let b1 = getDiscriminant(1, my, mx2, mx1, a11, a2, mx2, a22, a3) / getDiscriminant(1, mx1, mx2, mx1, a1, a2, mx2, a2, a3)
let b2 = getDiscriminant(1, mx1, my, mx1, a1, a11, mx2, a2, a22) / getDiscriminant(1, mx1, mx2, mx1, a1, a2, mx2, a2, a3)

let y_pr1 = b0 + b1 * normalize_x[0][0] + b2 * normalize_x[0][1]
let y_pr2 = b0 + b1 * normalize_x[1][0] + b2 * normalize_x[1][1]
let y_pr3 = b0 + b1 * normalize_x[2][0] + b2 * normalize_x[2][1]

let dx1 = Math.abs(x1Max - x1Min) / 2
let dx2 = Math.abs(x2Max - x2Min) / 2
let x10 = (x1Max + x1Min) / 2
let x20 = (x2Max + x2Min) / 2

let koef0 = b0 - (b1 * x10 / dx1) - (b2 * x20 / dx2)
let koef1 = b1 / dx1
let koef2 = b2 / dx2

let yP1 = koef0 + koef1 * x1Min + koef2 * x2Min
let yP2 = koef0 + koef1 * x1Max + koef2 * x2Min
let yP3 = koef0 + koef1 * x1Min + koef2 * x2Max

console.log(Y)
console.log("matrix")
for (let i = 0; i < 3; i++) {
    console.log(Y[i])
}
for (let i = 0; i < 3; i++) {
    console.log(R_uv[i])
}
console.log(Math.round(koef0, 4), Math.round(koef1, 4), Math.round(koef2, 4))
console.log(Math.round(y_pr1, 4), Math.round(y_pr2, 4), Math.round(y_pr3, 4))
console.log(Math.round(yP1, 4), Math.round(yP2, 4), Math.round(yP3, 4))
console.log(b0, b1, b2)