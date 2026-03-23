"""
Evaluator för neurala nätverk som predikterar responsfunktioner
som funktion av q och omega.

Kan användas för att:
- hämta sann data ur dataset
- göra prediktioner med ett givet NN
- utvärdera MSE för valda q-värden och funktioner
"""

class NNEvaluator:
    """
    Klass för att utvärdera neurala nätverk på responsfunktioner.

    Klassen jämför modellens prediktioner med sann data för valda
    q-värden och responsfunktioner.
    """    
    def __init__(self, dataset, function_order, prediction_fn, mask_small_values = False, small_value_threshold = 1e-8):
        self.dataset = dataset
        self.function_order = function_order
        self.prediction_fn = prediction_fn
        self.q_list = sorted(dataset.keys())
        self.mask_small_values = mask_small_values
        self.small_value_threshold = small_value_threshold

    def get_q_values(self):
        """Returnerar alla q-värden i datasetet.
        """

        return self.q_list.copy()
    
    def resolve_q_values(self, q_values = "all_valid"):
        """Tolkar q_values-argumentet och returnerar en lista av q-värden att utvärdera på.
        """
        if q_values == "all_valid":
            return self.q_list[1:-1] # Exkluderar första och sista q-värdet
        
        if isinstance(q_values, (int, float)):
            q_values = float(q_values)
            return [q_values]
        
        return list(q_values)
    
    def resolve_functions(self, functions = "all"):
        """Tolkar functions-argumentet och returnerar en lista av funktionsnamn att utvärdera på.
        """
        if functions == "all":
            return self.function_order.copy()
        
        if isinstance(functions, str):
            return [functions]
        
        return list(functions)
       
    def get_true_data(self, q_value, functions = "all"):
        """Hämtar sann data för ett givet q-värde och en lista av funktioner.
        """
        functions = self.resolve_functions(functions)
        omega = self.dataset[q_value]["omega"]

        true_data = {}
        for func in functions:
            true_data[func] = self.dataset[q_value][func]

        return omega, true_data
    
    def predict(self, model, q_value):
        """Gör prediktioner med modellen för ett givet q-värde.
        """
        omega = self.dataset[q_value]["omega"]

        return self.prediction_fn(model, q_value, omega)
    
    def mse(self, y_true, y_pred, omega = None):
        """Beräknar medelkvadratfelet mellan sann data och prediktioner.
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def evaluate(self, model, q_values = "all_valid", functions = "all", metric = "mse"):
        """
        Utvärderar modellen med {metric} för valda q-värden och funktioner.

        Parametrar:
        model: tränad modell som ska utvärderas
        q_values: ett q-värde, en lista av q-värden, eller "all_valid"
        functions: en funktion, en lista av funktioner, eller "all"
        metric: måttet som ska användas för utvärdering

        Returnerar:
        dict: resultat på formen results[q][function] = metric_value
        dict: genomsnittlig {metric} per funktion över alla q på formen fnc_avg[function] = metric_value
        float: genomsnittlig {metric} över alla q och funktioner 
        """
        if not hasattr(self, metric):
            raise ValueError(f"Metric '{metric}' är inte definierad i NNEvaluator.")
        
        metric_fn = getattr(self, metric)
        
        q_values = self.resolve_q_values(q_values)
        functions = self.resolve_functions(functions)


        results = {}
        for q in q_values:
            self.validate_predictions(model, q, functions)
            omega, true_data = self.get_true_data(q,functions)
            predictions = self.predict(model, q)
            results[q] = {}
            for func in true_data:
                metric_value = metric_fn(true_data[func], predictions[func], omega)
                results[q][func] = metric_value
        fnc_avg = {func: np.mean([results[q][func] for q in results]) for func in functions} # Genomsnittlig {metric} per funktion över alla q
        total_avg = np.mean([metric_value for q in results for metric_value in results[q].values()]) # Genomsnittlig {metric} över alla q och funktioner
        return results, fnc_avg, total_avg
    
    def validate_predictions(self, model, q_value, functions = "all"):
        """
        Säkerställer att prediktionerna och sann data är jämförbara i funktioner och shape.
        Används för att fånga potentiella problem innan utvärdering.
        """
        _, true_data = self.get_true_data(q_value, functions)
        predictions = self.predict(model, q_value)

        for func in true_data:
            if func not in predictions:
                raise ValueError(f"Funktionen '{func}' saknas i prediktionerna.")
            if true_data[func].shape != predictions[func].shape:
                raise ValueError(f"Formen på sann data och prediktioner för '{func}' matchar inte.")
            
    def integral_abs_error(self, y_true, y_pred, omega):
        """
        Beräknar det absoluta integrerade felet mellan sann data och prediktioner över omega.

        Parametrar:
        y_true: sann data som funktion av omega
        y_pred: prediktioner som funktion av omega
        omega: array av omega-värden

        Returnerar:
        float: det absoluta integrerade felet
        """
        return np.trapz(np.abs(y_true - y_pred), x=omega)     
    def rmse(self, y_true, y_pred, omega = None):
        """
        Beräknar rotmedelkvadratfelet mellan sann data och prediktioner.

        Parametrar:
        y_true: sann data som funktion av omega
        y_pred: prediktioner som funktion av omega
        omega: (valfritt) array av omega-värden, inte nödvändigt för RMSE

        Returnerar:
        float: rotmedelkvadratfelet
        """

        if self.mask_small_values:
            mask = (np.abs(y_true) >= self.small_value_threshold)
            if not np.any(mask):
                raise ValueError("Alla värden i y_true är under small_value_threshold. Justera tröskeln eller inaktivera maskering.")

            y_true = y_true[mask]
            y_pred = y_pred[mask]       


        return np.sqrt(np.mean((y_true - y_pred) ** 2))   
    
    def get_peak_info(self, y, omega): 
        """
        Hittar positionen och höjden av den största toppen i responsfunktionen.

        Parametrar:
        y: responsfunktion som funktion av omega
        omega: array av omega-värden

        Returnerar:
        tuple: (peak_position, peak_height)
        """
        peak_index = np.argmax(y)
        peak_position = omega[peak_index]
        peak_height = y[peak_index]
        return peak_position, peak_height

    def peak_position_error(self, y_true, y_pred, omega):
        """
        Beräknar felet i positionen av den största toppen i responsfunktionen.

        Parametrar:
        y_true: sann data som funktion av omega
        y_pred: prediktioner som funktion av omega
        omega: array av omega-värden

        Returnerar:
        float: felet i positionen av den största toppen
        """

        true_peak_position, _ = self.get_peak_info(y_true, omega)
        pred_peak_position, _ = self.get_peak_info(y_pred, omega)
        return np.abs(true_peak_position - pred_peak_position)
    
    def peak_height_error(self, y_true, y_pred, omega):
        """
        Beräknar felet i höjden av den största toppen i responsfunktionen.

        Parametrar:
        y_true: sann data som funktion av omega
        y_pred: prediktioner som funktion av omega
        omega: array av omega-värden

        Returnerar:
        float: felet i höjden av den största toppen
        """
        _, true_peak_height = self.get_peak_info(y_true, omega)
        _, pred_peak_height = self.get_peak_info(y_pred, omega)
        return np.abs(true_peak_height - pred_peak_height)

    def plot_prediction(self, model, q_value, functions = "all", show_peaks = False):
        """
        Plottar sann data och modellens prediktioner för ett givet q-värde och funktioner.

        Parametrar:
        model: tränad modell som ska utvärderas
        q_value: det q-värde som ska plottas
        functions: en funktion, en lista av funktioner, eller "all"
        show_peaks: om True, markera positionen och höjden av den största toppen i plottarna
        """

        import matplotlib.pyplot as plt

        functions = self.resolve_functions(functions)
        self.validate_predictions(model, q_value, functions)
        omega, true_data = self.get_true_data(q_value, functions)
        predictions = self.predict(model, q_value)

        for func in functions:
            plt.figure(figsize=(10, 6))
            plt.plot(omega, true_data[func], label="Sann data", color="blue")
            plt.plot(omega, predictions[func], label="Prediktion", color="orange", linestyle="--")
            plt.title(f"Responsfunktion '{func}' för q={q_value}")
            plt.xlabel("Omega")
            plt.ylabel(func)
            plt.grid()

            if show_peaks:
                true_peak_pos, true_peak_height = self.get_peak_info(true_data[func], omega)
                pred_peak_pos, pred_peak_height = self.get_peak_info(predictions[func], omega)
                plt.scatter([true_peak_pos], [true_peak_height], color="red", marker="o", label="Sann topp")
                plt.scatter([pred_peak_pos], [pred_peak_height], color="green", marker="x", label="Predikterad topp")
                
            plt.legend()    
            plt.show()
        

