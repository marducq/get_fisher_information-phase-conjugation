import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import special
import random

from fisher_functions import get_W3
import copy
import torch
random.seed(42)

class DipoleSimulation:
    def __init__(self, Nb_particule=100, W=3, L=1, k_in=25, N_x=100, N_y=300  ):
        self.Nb_particule = Nb_particule  # nombre de dipôles
        self.W = W                        # largeur du milieu
        self.L = L                        # longueur du milieu
        self.k_in = k_in                  # nombre d'onde incident
        self.N_x = N_x                    # résolution de la grille pour les particules
        self.N_y = N_y
        self.alpha = 4 * 1j / (k_in**2)
        self.pas_x = L / N_x
        self.pas_y = W / N_y
        self.ksi = L / N_y
        self.Nb_input_modes = 30
        # Génère la configuration initiale des dipôles
        self.X_random, self.Y_random = self.creation_grille_avec_points_aleat()
        # Construit la matrice d'interaction A entre dipôles
        self.A = self.compute_interaction_matrix()
        
    def creation_grille_avec_points_aleat(self):
        
        #Création d'une grille sur [0,L]x[0,W] et sélection aléatoire de Nb_particule positions.
        
        x = np.linspace(0, self.L, self.N_x)
        y = np.linspace(0, self.W, self.N_y)
        X, Y = np.meshgrid(x, y)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        total_points = len(X_flat)
        indices = np.random.choice(total_points, self.Nb_particule, replace=False)
        X_random = X_flat[indices]
        Y_random = Y_flat[indices]
        return X_random, Y_random

    def green_function(self, r, r_prime, k0, eps=1e-9):
        R = np.linalg.norm(np.array(r) - np.array(r_prime))
        if R < eps:
            return 0
        return 1j * special.hankel1(0, k0 * R) / 4

    def champ_incident(self, x, y, thet):
        
        return np.exp(1j * self.k_in * (np.cos(thet)*x + np.sin(thet)*y))
    
    def compute_interaction_matrix(self):
        
        A = np.zeros((self.Nb_particule, self.Nb_particule), dtype=complex)
        for j in range(self.Nb_particule):
            for k in range(self.Nb_particule):
                if j == k:
                    A[j, k] = 0
                else:
                    A[j, k] = self.k_in**2 * self.alpha * self.green_function(
                        (self.X_random[j], self.Y_random[j]),
                        (self.X_random[k], self.Y_random[k]),
                        self.k_in)
        return A
    
    def compute_field(self, thet, N_obs_x=100, N_obs_y=30):
        
        # Calcul du champ incident sur chaque dipôle
        E0 = np.array([self.champ_incident(x, y, thet) 
                       for x, y in zip(self.X_random, self.Y_random)])
        I_mat = np.eye(self.Nb_particule, dtype=complex)
        M = I_mat - self.A
        E = np.linalg.solve(M, E0)
        
        # Définition de la grille d'observation (pour couvrir la zone réfléchie et transmise)
        x_obs = np.linspace(-self.L, 2*self.L, N_obs_x)
        y_obs = np.linspace(-self.L, self.W+self.L, N_obs_y)
        X_obs, Y_obs = np.meshgrid(x_obs, y_obs)
        E_total = np.zeros_like(X_obs, dtype=complex)
        
        # Calcul du champ total point par point
        for i in range(N_obs_y):
            for j in range(N_obs_x):
                r_obs = (X_obs[i, j], Y_obs[i, j])
                E0_r = self.champ_incident(r_obs[0], r_obs[1], thet)
                somme = 0.0 + 0.0j
                for idx in range(self.Nb_particule):
                    r_part = (self.X_random[idx], self.Y_random[idx])
                    somme += self.green_function(r_obs, r_part, self.k_in) * E[idx]
                E_total[i, j] = E0_r + self.k_in**2 * self.alpha * somme
        return x_obs, y_obs, E_total

    def build_TM(self, theta_vals, N_obs_x=100, N_obs_y=30):
        
        #Pour chaque angle, extrait le champ en sortie (à x=2L) 
        #et construit la matrice de transmission pré-transposée.
        
        TM_pretransposee = []
        for idx, th in enumerate(theta_vals):
            progress = (idx+1) / len(theta_vals) * 100
            print(f"Traitement de theta {idx+1}/{len(theta_vals)} : {progress:.1f}% terminé")
            x_obs, _, E_total = self.compute_field(th, N_obs_x, N_obs_y)
            # Extraction de la colonne à x = 2L
            indx = np.argmin(np.abs(x_obs - 2*self.L))
            field_line = E_total[:, indx]
            TM_pretransposee.append(field_line)
        return np.array(TM_pretransposee)  # forme (N_input, N_obs_y)
    

    def deplacement_particule(self, index, ksi, update_matrix=True):
        
        
        # Appliquer le déplacement sur la coordonnée Y de la particule choisie
        self.Y_random[index] += ksi
        
        # Optionnellement, mettre à jour la matrice d'interaction si les positions ont changé
        if update_matrix:
            self.A = self.compute_interaction_matrix()

    
    def deplacement_particules_random(self, n=10 , update_matrix=True):
        
        #Déplace aléatoirement n particules de la configuration actuelle
        #d'une distance self.ksi en Y sur la grille.

        #Paramètres :
        #- n : int, nombre de particules à déplacer.
        #- update_matrix : bool, si True (par défaut), met à jour la matrice d'interaction après déplacement.

        #Exemple :
        
        # Déplacer aléatoirement 5 particules de self.ksi en Y
        #sim.deplacement_particules_random(5)
        
        
        if not isinstance(n, int) or n < 1:
            raise ValueError("Le nombre de particules à déplacer doit être un entier positif")
        if n > self.Nb_particule:
            raise ValueError("n ne peut pas dépasser le nombre total de particules")

        # Sélection aléatoire d'indices uniques
        indices = np.random.choice(self.Nb_particule, size=n, replace=False)

        # Appliquer le déplacement standard self.ksi à chaque particule sélectionnée
        for idx in indices:
            self.Y_random[idx] += self.ksi

        # Mettre à jour la matrice d'interaction si demandé
        if update_matrix:
            self.A = self.compute_interaction_matrix()

    



    def compute_line_field(self, thet, N_obs_x=100, N_obs_y=30):
        # calcule le champ total mais uniquement sur la ligne x=2L
        # Définition de la grille d'observation
        x_obs = np.linspace(-self.L, 2*self.L, N_obs_x)
        y_obs = np.linspace(-self.L, self.W+self.L, N_obs_y)
        # Trouver l'indice correspondant à x = 2L
        indx = np.argmin(np.abs(x_obs - 2*self.L))
        
        # Calcul du champ incident sur chaque dipôle et résolution du système
        E0 = np.array([self.champ_incident(x, y, thet) 
                       for x, y in zip(self.X_random, self.Y_random)])
        I_mat = np.eye(self.Nb_particule, dtype=complex)
        M = I_mat - self.A
        E = np.linalg.solve(M, E0)
        
        # Calcul du champ uniquement pour la colonne correspondant à x = 2L
        E_line = np.zeros(N_obs_y, dtype=complex)
        for i in range(N_obs_y):
            r_obs = (x_obs[indx], y_obs[i])
            E0_r = self.champ_incident(r_obs[0], r_obs[1], thet)
            somme = 0.0 + 0.0j
            for idx in range(self.Nb_particule):
                r_part = (self.X_random[idx], self.Y_random[idx])
                somme += self.green_function(r_obs, r_part, self.k_in) * E[idx]
            E_line[i] = E0_r + self.k_in**2 * self.alpha * somme
        return y_obs, E_line
    
    def intensity_difference(self, thet, N_obs_x, N_obs_y, ksi):
        """
        Calcule la différence (norme L2) entre l'intensité d'une configuration de référence
        et celle obtenue après avoir déplacé une particule de valeur 'ksi'.
        """
        # Configuration de référence
        x_obs, y_obs, E_total_ref = self.compute_field(thet, N_obs_x, N_obs_y)
        I_ref = np.abs(E_total_ref)**2
        
        # Créer une copie profonde pour appliquer le déplacement
        import copy
        sim_mod = copy.deepcopy(self)
        sim_mod.ksi = ksi
        sim_mod.deplacement_1particule()
        _, _, E_total_mod = sim_mod.compute_field(thet, N_obs_x, N_obs_y)
        I_mod = np.abs(E_total_mod)**2
        
        diff = np.linalg.norm(I_mod - I_ref)
        return diff
    
    def compute_input_field_particlebasis(self, input_field, theta_vals):
        
        #Transforme un vecteur d'entrée optimisé Xopt, défini dans un espace réduit de dimension
        #Nb_input_modes, en un vecteur d'entrée complet de dimension self.Nb_particule, en supposant que 
        #chaque composante de Xopt correspond à une onde plane incidente depuis un angle donné (dans theta_vals).

        #Paramètres:
        #  Xopt       : Vecteur optimisé (dimension Nb_input_modes).
        # theta_vals : Tableau des angles associés aux modes (taille Nb_input_modes).

        #Retourne:
        #  input_field_full : Vecteur d'entrée complet qui correspond au champ incident percu par chacune des particules ( cest E_0( r_j) du TD ) (dimension self.Nb_particule).
        
        E0 = np.zeros(self.Nb_particule, dtype=complex)
        for j in range(self.Nb_particule):
            x_j = self.X_random[j]
            y_j = self.Y_random[j]
            inc = 0.0 + 0.0j
            for n in range(len(theta_vals)):
                # Superposition pondérée d'ondes planes pour créer le champ incident en (x_j, y_j)
                inc += input_field[n] * np.exp(1j * self.k_in * (np.cos(theta_vals[n]) * x_j + np.sin(theta_vals[n]) * y_j))
            E0[j] = inc
        return E0
        


    def compute_field_with_input(self, theta_vals, N_obs_x, N_obs_y, input_field):
        """
        Calcule le champ total E_total sur la grille d'observation en utilisant
        un vecteur d'entrée custom (input_field) pour exciter les dipôles.

        Paramètres :
        theta_vals  : Angles d'incidence (radians).
        N_obs_x     : Nombre de points dans la grille d'observation en x.
        N_obs_y     : Nombre de points dans la grille d'observation en y.
        input_field : Vecteur d'entrée imposé sur les dipôles (dimension = Nb_input_modes).

        Retourne :
        x_obs       : Grille en x (1D).
        y_obs       : Grille en y (1D).
        E_total     : Champ total complexe (2D de dimensions (N_obs_y, N_obs_x)).
        """
        # On utilise directement le vecteur input_field pour exciter les dipôles.
        E0 = self.compute_input_field_particlebasis(input_field, theta_vals)  # Dimension attendue : (self.Nb_particule,)
       

        # Résolution du système linéaire (I - A)E = E0 afin d'obtenir la réponse des dipôles E.
        I_mat = np.eye(self.Nb_particule, dtype=complex)
        M = I_mat - self.A
        E = np.linalg.solve(M, E0)

        # Définition de la grille d'observation
        x_obs = np.linspace(-self.L, 2 * self.L, N_obs_x)
        y_obs = np.linspace(-self.L, self.W + self.L, N_obs_y)
        X_obs, Y_obs = np.meshgrid(x_obs, y_obs)

        # Initialisation du champ total sur la grille
        E_total = np.zeros_like(X_obs, dtype=complex)
        for i in range(N_obs_y):
            for j in range(N_obs_x):
                # Coordonnées d'observation
                r_obs = (X_obs[i, j], Y_obs[i, j])
                contributio = 0.0 + 0.0j

                # Champ incident en ce point
                if len(input_field) != len(theta_vals):
                    raise ValueError("Mismatch between input_field length and theta_vals length.")
                for n in range(len(theta_vals)):
                    contributio += input_field[n] * np.exp(
                        1j * self.k_in * (np.cos(theta_vals[n]) * r_obs[0] + np.sin(theta_vals[n]) * r_obs[1])
                    )

                E0_r = contributio

                # Contribution de chaque dipôle via la fonction de Green
                somme = 0.0 + 0.0j
                for idx in range(self.Nb_particule):
                    r_part = (self.X_random[idx], self.Y_random[idx])
                    somme += self.green_function(r_obs, r_part, self.k_in) * E[idx]

                # Composition du champ total (champ incident + contribution des dipôles)
                E_total[i, j] = E0_r + self.k_in**2 * self.alpha * somme

        return x_obs, y_obs, E_total


    def compute_Q(self, theta_vals, N_obs_x, N_obs_y, particle_index=50, ksi_calib=None):
        
        TM_ref, TM_calib = self.compute_TMs_pair(theta_vals, N_obs_x, N_obs_y,
                                                 particle_index=particle_index,
                                                 ksi_calib=ksi_calib)
        # dH : différence finie entre la TM calibrée et la TM de référence
        dH = (TM_calib - TM_ref) / (ksi_calib if ksi_calib is not None else self.L/self.N_y)
        Q = np.conjugate(dH).T @ dH
        return TM_ref , TM_calib, Q

    def compute_TMs_pair(self, theta_vals, N_obs_x, N_obs_y, particle_index=50, ksi_calib=None):
    
        # Si aucun déplacement n'est fourni, on le définit par défaut
        if ksi_calib is None:
            ksi_calib = self.L / self.N_y

        # Calcul de la TM de référence :
        # La méthode build_TM boucle sur theta_vals et renvoie un tableau de forme (Nb_input_modes, N_obs_y).
        TM_ref_all = self.build_TM_surdetecteur( theta_vals, N_obs_x , N_obs_y=300 , particle_index = 50 , ksi_calib = None)
        # Transposer pour obtenir une TM de forme (Nb_output_modes, Nb_input_modes)
        TM_ref = TM_ref_all.T

        # Calcul de la TM de calibration :
        sim_calib = copy.deepcopy(self)

        # si on veut déplacer une particule à la fois
        #sim_calib.deplacement_particule(index=particle_index, ksi=ksi_calib)


        # si on veut déplacer plusieurs particules à la fois 
        sim_calib.deplacement_particules_random(n=10, update_matrix=True)


        TM_calib_all = sim_calib.build_TM_surdetecteur(theta_vals, N_obs_x , N_obs_y=300 , particle_index = 50 , ksi_calib = None)
        TM_calib = TM_calib_all.T

        return TM_ref, TM_calib

    def build_TM_surdetecteur(self, theta_vals, N_obs_x , N_obs_y=300 , particle_index = 50 , ksi_calib = None):
        """
        Pour chaque angle de theta_vals, calcule le champ total au détecteur
        situé à x = 1.5 pour y ∈ [1, 2] en N_obs_y points.
        Renvoie un tableau de forme (len(theta_vals), N_obs_y).
        """
        x_det = 1.5
        y_min, y_max = 1.0, 2.0

        # grille du détecteur en y
        y_obs = np.linspace(y_min, y_max, N_obs_y)

        TM_pretransposee = []

        # matrice I - A (constante si la configuration de dipôles ne change pas)
        M = np.eye(self.Nb_particule, dtype=complex) - self.A

        for idx, th in enumerate(theta_vals):
            print(f"Traitement de theta {idx+1}/{len(theta_vals)}")

            # 1) Calcul du champ incident sur chaque dipôle et résolution de (I-A)E = E0
            E0 = np.array([self.champ_incident(xj, yj, th)
                        for xj, yj in zip(self.X_random, self.Y_random)])
            E = np.linalg.solve(M, E0)

            # 2) Calcul du champ sur le détecteur
            field_line = np.zeros(N_obs_y, dtype=complex)
            for i, y in enumerate(y_obs):
                r_obs = (x_det, y)
                # champ incident au détecteur
                E0_r = self.champ_incident(x_det, y, th)
                # contribution des dipôles
                somme = 0+0j
                for j in range(self.Nb_particule):
                    r_part = (self.X_random[j], self.Y_random[j])
                    somme += self.green_function(r_obs, r_part, self.k_in) * E[j]
                # champ total
                field_line[i] = E0_r + (self.k_in**2 * self.alpha) * somme

            TM_pretransposee.append(field_line)

        # résultat de forme (N_input, N_obs_y)
        return np.array(TM_pretransposee)


        








class PhaseConjugation:
    def __init__(self, TM_pretransposee):
        
        self.TM_pretransposee = TM_pretransposee
        # Calcul de l'adjoint (conjugué-transposé) de TM_pretransposee
        self.TM_dagger = TM_pretransposee.conj()  # forme (N_obs_y, N_input)
        # Pour la propagation, nous définissons la TM dans la convention (N_obs_y, N_input)
        self.TM = TM_pretransposee.T
        print("TM shape = " , self.TM.shape)

    def compute_input_field(self, desired_output):
        
        return np.dot(self.TM_dagger, desired_output)

    def phase_only(self, input_field):
        
        return np.exp(1j * np.angle(input_field))

    def compute_focused_output(self, phase_input):
        
        return np.dot(self.TM, phase_input)
    

    def phaseconj_ampl_phase(self, desired_output, normalize=True):
        """
        Calcule la solution amplitude+phase x_ls = H^+ · desired_output,
        avec H^+ la pseudo‑inverse de H.

        Si normalize=True, on renormalise ||x_ls||₂ = 1 pour conserver 
        la même « puissance totale ».
        """
        # 1) pseudo‑inverse de H
        H = self.TM                  # shape (N_output, N_input)
        x_ls = np.linalg.pinv(H) @ desired_output

        # 2) normalisation L2 optionnelle
        if normalize:
            x_ls = x_ls / np.linalg.norm(x_ls)

        return x_ls


