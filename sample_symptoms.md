# Cas de test pour l'Agent de Triage Médical (CHSA)

Voici 5 cas typiques couvrant les trois niveaux d'urgence (`max`, `moderate`, `deferred`) pour tester la robustesse de l'API.

> **💡 Note :** Les commandes ci-dessous sont sur une seule ligne pour faciliter le copier-coller.

## 1. Urgence Vitale (Cardiaque)
**Symptômes :** "Douleur thoracique brutale, comme un étau, qui irradie dans le bras gauche et la mâchoire. J'ai du mal à respirer et je suis couvert de sueurs froides depuis 10 minutes."
**Commande :**
```bash
make gcp-triage-pretty SYMPTOMS="Douleur thoracique brutale, comme un étau, qui irradie dans le bras gauche et la mâchoire. J'ai du mal à respirer et je suis couvert de sueurs froides depuis 10 minutes."
```

## 2. Urgence Neurologique (AVC)
**Symptômes :** "Ma mère a soudainement la bouche de travers et elle n'arrive plus à lever son bras droit. Ses propos sont incohérents et elle semble confuse."
**Commande :**
```bash
make gcp-triage-pretty SYMPTOMS="Ma mère a soudainement la bouche de travers et elle n'arrive plus à lever son bras droit. Ses propos sont incohérents et elle semble confuse."
```

## 3. Urgence Modérée (Traumatologie)
**Symptômes :** "Je suis tombé dans les escaliers il y a deux heures. Ma cheville est très gonflée et bleue, je n'arrive plus du tout à poser le pied par terre, la douleur est vive."
**Commande :**
```bash
make gcp-triage-pretty SYMPTOMS="Je suis tombé dans les escaliers il y a deux heures. Ma cheville est très gonflée et bleue, je n'arrive plus du tout à poser le pied par terre, la douleur est vive."
```

## 4. Urgence Différée (Infection mineure)
**Symptômes :** "J'ai un peu de fièvre (38.2°C) et le nez qui coule depuis hier. J'ai aussi mal à la gorge quand j'avale, mais je respire normalement."
**Commande :**
```bash
make gcp-triage-pretty SYMPTOMS="J'ai un peu de fièvre (38.2°C) et le nez qui coule depuis hier. J'ai aussi mal à la gorge quand j'avale, mais je respire normalement."
```

## 5. Cas de Routine (Prévention)
**Symptômes :** "Je souhaiterais obtenir un certificat médical pour reprendre le sport en club. Je n'ai aucune douleur particulière, c'est juste pour un bilan de contrôle annuel."
**Commande :**
```bash
make gcp-triage-pretty SYMPTOMS="Je souhaiterais obtenir un certificat médical pour reprendre le sport en club. Je n'ai aucune douleur particulière, c'est juste pour un bilan de contrôle annuel."
```
