const email = "codecamp@gmail.com"
// undefined
email.includes('@')
// true
email.split('@')
// (2) ['codecamp', 'gmail.com']
let userMail = email.split('@')[0]
// undefined
let company = email.split('@')[1]
// undefined
let maskingMail = []
// undefined
maskingMail.push(userMail[0])
// 1
maskingMail.push(userMail[1])
// 2
maskingMail.push(userMail[2])
// 3
maskingMail.push(userMail[3])
// 4
maskingMail.push('*')
// 5
maskingMail.push('*')
// 6
maskingMail.push('*')
// 7
maskingMail.push('*')
// 8
maskingMail
// (8) ['c', 'o', 'd', 'e', '*', '*', '*', '*']
maskingMail.join("")+'@'+company
// 'code****@gmail.com'
let result =maskingMail.join("")+'@'+company
// undefined
result
// 'code****@gmail.com'