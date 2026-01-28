from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from .forms import RegisterForm, LoginForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm

def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()  # le mot de passe est hashé automatiquement
            login(request, user)  # on connecte l'utilisateur
            return redirect('core:dashboard')  # home_user_view

    else:
        form = RegisterForm()
    return render(request, 'acountes/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)

            print("LOGIN OK")
            print("USER:", request.user)
            print("AUTH:", request.user.is_authenticated)

            return redirect('core:dashboard')  # home_user_view

    else:
        form = LoginForm()

    return render(request, 'acountes/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')
@login_required
def profile_view(request):
    print("METHOD:", request.method)
    return render(request, 'acountes/profile.html')




