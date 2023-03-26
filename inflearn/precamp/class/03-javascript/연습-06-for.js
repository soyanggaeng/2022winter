let persons = [
    {name: "철수", age: 17},
    {name: "영희", age: 22},
    {name: "도우너", age: 5},
    {name: "그루트", age: 65},
    {name: "도비", age: 3}
    ]
// undefined
for(let count = 0; count < persons.length; count++){
    if (persons[count].age >= 19){
        console.log(persons[count].name + " 님은 성인입니다")
    }
    else {
        console.log(persons[count].name + "님은 미성년자입니다")
    }
}
// VM2497:6 철수님은 미성년자입니다
// VM2497:3 영희 님은 성인입니다
// VM2497:6 도우너님은 미성년자입니다
// VM2497:3 그루트 님은 성인입니다
// VM2497:6 도비님은 미성년자입니다