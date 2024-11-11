function showpwd() {
    console.log('hiigfgdffjgednj irjgnsdojf jdok')
    const pwd = document.getElementById('password');
    const pwdbtn = document.getElementById('pwdshow');
    if (pwd.type = "password") {
        pwd.type = "text";
        pwdbtn.innertext = 'Hide Password';
    } 
    else {
        pwd.type = "password";
        pwdbtn.innertext = 'Show Password';
    }
}
