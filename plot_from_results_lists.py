import matplotlib
matplotlib.use('Qt5Agg')

import numpy
import matplotlib.pyplot as plt


def plot_my_results(output_dir, dataset):
    colors = ['b','g','r','c','m','y','k', 'dimgray']
    colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    # bank dataset
    font_size= 14
    size = 8
    if dataset == "bank":

        names = ['Zafar et al.', 'Krasanakis et al.', 'AdaBoost', 'AdaFair']

        accuracy_list = [0.9014654090146542,0.8986906 , 0.8997750224977501,.87381761823817 ]
        std_accuracy_list = [0.0020856491649241297,0.000369014633856, 0.002307587474996224,.012432088]
        balanced_accuracy_list =[0.6569178324659456, 0.64968265, 0.6884338555612595, 0.786019 ]
        std_balanced_accuracy_list =[0.004124698886348359, 0.003130033274584, 0.006112057444927729, .020544993]
        fairness_list =[0.026247620510718384, 0.029363, 0.11459964385212498, .0305449 ]
        std_fairness_list =[0.013302190089989351, 0.01322239448814, 0.039373123904823705, 0.022990 ]

        tpr_protected_list =[0.3404819694085027, 0.3399602, 0.3769288688278519, 0.6737057 ]
        std_tpr_protected_list= [0.013168271097689893, 0.027213980849924, 0.019418388890897493, 0.04565]
        tpr_non_protected_list =[0.33533764082441775, 0.3228484, 0.46463747980275044, .668128 ]
        std_tpr_non_protected_list= [0.0170381072064758, 0.048545932917187, 0.025794234314879837, 0.043146 ]
        tnr_protected_list= [0.9772712011898972, 0.969662, 0.96924162038161381, 0.9073496467 ]
        std_tnr_protected_list =[0.0010575422085822227, 0.005373073189526, 0.002439124361957216, 0.021830 ]
        tnr_non_protected_list =[0.9715604016454734, 0.9798354, 0.9455251709389116,0.885394232914 ]
        std_tnr_non_protected_list= [0.002640862219941835,0.005692142153882, 0.00556099856681456, 0.0238926]

    elif dataset == "compass":
        names = ['Zafar et al.', 'Krasanakis et al.', 'AdaBoost', 'AdaFair']

        accuracy_list =[0.6442970822281168, 0.6709576, 0.6712769988632057, .6471011746]
        std_accuracy_list =[0.005771835723483403,0.002795755050071, 0.010489284225590305, 0.010058541]
        balanced_accuracy_list = [0.6357560937166155, 0.6594696875,0.6675175086392817,.654489 ]
        std_balanced_accuracy_list = [0.006220496647839287, 0.002896127500646,0.010979180765098788, 0.0105]
        fairness_list =[0.09359745167922937, 0.0424834, 0.3612745751493023, 0.085852]
        std_fairness_list =[0.028390909071829718, 0.019471642566238, 0.11452832362515976, .04884]

        tpr_protected_list =[0.4549621658317228, 0.5501926, 0.41987127961375315, 0.65728]
        std_tpr_protected_list=[0.0393259444451863, 0.112697835143546, 0.06886461753830742,0.05791 ]
        tpr_non_protected_list =[0.49193300734626444, 0.5724758, 0.6374989253709892, 0.675284]
        std_tpr_non_protected_list=[0.030444830417997913, 0.13459096214682, 0.01740228200829841, 0.043681 ]
        tnr_protected_list=[0.7560185338901767, 0.7689726,  0.8408247749911053, 0.60997]
        std_tnr_protected_list =[0.032014630828735395, 0.09271311259059, 0.043274616827514024, 0.02619]
        tnr_non_protected_list =[0.7935183026247083, 0.76223775, 0.6971778455990392, .62879051]
        std_tnr_non_protected_list=[0.03743195061831913, 0.10889897263139, 0.02245981608568674, 0.035784]



    elif dataset == "adult":
        names = ['Zafar et al.', 'Krasanakis et al.', 'AdaBoost', 'AdaFair']

        accuracy_list =[0.8353046936807845, 0.8272768, 0.8475572663022305, 0.8334735257658934]
        std_accuracy_list =[0.0013164209815277018, 0.00033013739564, 0.0016006291589892912, 0.00890679143447399 ]
        balanced_accuracy_list = [0.7096840514775167, 0.70314485, 0.7336316139467047, 0.7828964294538823]
        std_balanced_accuracy_list =[0.0041676454503743595, 0.002760948261015,  0.0045062591280023954, 0.009487467261529667]
        fairness_list = [0.13379807768266622, 0.051112,0.2228647743568891, .083534562456318]
        std_fairness_list =[0.009987622284623092, 0.009158556474685, 0.017486347288878254, 0.0203518118]

        tpr_protected_list =[0.5591357574100587, 0.4361816, 0.3625298364804249, .720928125]
        std_tpr_protected_list= [0.013522423527874023, 0.035273984476881, 0.018062454772540468, 0.0573654]
        tpr_non_protected_list =[0.4427906596319713, 0.4565824,0.5330325825505013,0.675974339 ]
        std_tpr_non_protected_list= [0.010025543102808662, 0.049284699149844, 0.011479745452822406, 0.04230408]
        tnr_protected_list=[0.970002907720055, 0.9436514, 0.9919610963346415,  0.906922]
        std_tnr_protected_list =[0.0010566979231614421, 0.02552339920701, 0.001037263342282705, 0.022242]
        tnr_non_protected_list = [0.9525499278154762, 0.976164,0.9395990680478288, 0.8683419375]
        std_tnr_non_protected_list=[0.003620189603987732, 0.007641821041732,0.0038272631907209716, 0.02909538]


    elif dataset == "kdd":
        names = ['AdaBoost', 'AdaFair']

        accuracy_list = [0.9510608581757918, 0.8653020856304672]
        std_accuracy_list =[0.000403011679569436, 0.009482596863320694]
        balanced_accuracy_list= [ 0.6671950236883214, 0.8450236983993061]
        std_balanced_accuracy_list =  [0.0031339775011954506, 0.0016358671824901515]
        fairness_list=  [0.282380954017813, 0.0251882142186136]
        std_fairness_list = [0.01181685740347354, 0.009595585173706027]
        tpr_protected_list= [ 0.13170866168306072, 0.8242686431935535]
        std_tpr_protected_list = [ 0.007100406699076609, 0.02345361698685365]
        tpr_non_protected_list = [0.4000030727538516, 0.8212118652972811]
        std_tpr_non_protected_list = [0.007807929267985162, 0.010756865637867031]
        tnr_protected_list = [0.997993973529316, 0.87299393692887]
        std_tnr_protected_list = [0.0001707044931205857, 0.01331858107908297]
        tnr_non_protected_list = [0.9839074305822939, 0.86253327839291]
        std_tnr_non_protected_list = [ 0.0012323005001142508, 0.0085747375]

        colors = [ "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    elif dataset == "single_adult":
        names = ['AdaFair NoAccumFair', 'AdaFair']
        size = 5
        font_size = 10
        colors = [ "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        accuracy_list = [0.7777090014159209, 0.8363055161974137]
        std_accuracy_list = [0.030064684417580895, 0.007616864832906049]
        balanced_accuracy_list = [0.7987636933979619, 0.7764257552606315]
        std_balanced_accuracy_list = [0.0074068468263909235, 0.011653028874518221]
        fairness_list = [0.6166000122397814, 0.07457158255107559]
        std_fairness_list = [0.1579182551466377, 0.01710448303596057]
        tpr_protected_list = [0.5899305758936734, 0.671100283177135]
        std_tpr_protected_list = [0.042497513537486646, 0.06156457893754373]
        tpr_non_protected_list = [0.8841626189703365, 0.6551453361125438]
        std_tpr_non_protected_list = [0.07939441695204563, 0.04593464806413498]
        tnr_protected_list = [0.9560829603525566, 0.9268545355760752]
        std_tnr_protected_list = [0.01124469775573034, 0.023874255078996088]
        tnr_non_protected_list = [0.6337149911894382, 0.8757468271997537]
        std_tnr_non_protected_list = [0.09829222263484681, 0.027108668687253448]


    elif dataset == "single_compass":
        names = ['AdaFair NoAccumFair', 'AdaFair']
        size = 5
        font_size = 10
        colors = [ "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        accuracy_list = [0.6686813186813187, 0.6531072375899962]
        std_accuracy_list = [0.007496020990294776, 0.01355678435872223]
        balanced_accuracy_list = [0.6658623366276265, 0.6546584411236644]
        std_balanced_accuracy_list = [0.007604180285551037, 0.011478169183096694]
        fairness_list = [0.2272840304128841, 0.07633060290232009]
        std_fairness_list = [0.05506351872314813, 0.0490507351376023]
        tpr_protected_list = [0.5069406199462202, 0.6656682802275413]
        std_tpr_protected_list = [0.0381354508039561, 0.05032129369256071]
        tpr_non_protected_list = [0.6398830105321432, 0.6831817894682468]
        std_tpr_non_protected_list = [0.03720199511028842, 0.0606794692145965]
        tnr_protected_list = [0.7839015673753595, 0.622210869081045]
        std_tnr_protected_list = [0.02934122119890074, 0.07149642956701875]
        tnr_non_protected_list = [0.6895599275483985, 0.6308168614291902]
        std_tnr_non_protected_list = [0.03305301207796616, 0.06827488636484708]


    elif dataset == "single_bank":
        names = ['AdaFair NoAccumFair', 'AdaFair']
        font_size = 10
        size = 5
        colors = [ "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        accuracy_list = [0.8085591440855915, 0.8781796820317969]
        std_accuracy_list = [0.02354999414032148, 0.014524208278431245]
        balanced_accuracy_list = [0.8169505526843439, 0.7737190583478951]
        std_balanced_accuracy_list = [0.005353024009187448, 0.02525125603579959]
        fairness_list = [0.2059109064276321, 0.03825268211747555]
        std_fairness_list = [0.09393951064856917, 0.011628147572499576]
        tpr_protected_list = [0.7960659163308293, 0.6407251685920573]
        std_tpr_protected_list = [0.03396094265151269, 0.07540275212550052]
        tpr_non_protected_list = [0.873800019926338, 0.632672802445631]
        std_tpr_non_protected_list = [0.049172836565705626, 0.07513803706415173]
        tnr_protected_list = [0.8453768252367302, 0.9161547662845739]
        std_tnr_protected_list = [0.019853402510626564, 0.02494102449578215]
        tnr_non_protected_list = [0.7172000224046069, 0.8961132038093191]
        std_tnr_non_protected_list = [0.069610195263599, 0.02788842925696522]


    elif dataset == "single_kdd":
        font_size = 10

        names = ['AdaFair NoAccumFair', 'AdaFair']
        size = 5
        colors = [ "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]



    plt.figure(figsize=(size, size))
    plt.rcParams.update({'font.size': font_size})
    plt.ylim([0,1])
    plt.yticks(numpy.arange(0, 1, step=0.05))

    plt.setp(plt.gca().get_xticklabels(), rotation=20, horizontalalignment='right')

    plt.grid(True, axis='y')
    index = numpy.arange(0, 8, step=1.3)
    # index = numpy.arange(7)
    bar_width = 0.21



    plt.xticks(index + 1.5*bar_width ,
               ('Accuracy', 'Balanced Accuracy', 'Equalized Odds', 'TPR Prot.', 'TPR Non-Prot.', 'TNR Prot.', 'TNR Non-Prot.'))

    for i in range(0, len(names)):
        plt.bar(index + bar_width * i,
                [accuracy_list[i], balanced_accuracy_list[i], fairness_list[i], tpr_protected_list[i],
                 tpr_non_protected_list[i], tnr_protected_list[i], tnr_non_protected_list[i]], bar_width,
                yerr=[std_accuracy_list[i], std_balanced_accuracy_list[i], std_fairness_list[i],
                      std_tpr_protected_list[i], std_tpr_non_protected_list[i], std_tnr_protected_list[i],
                      std_tnr_non_protected_list[i]],
                label=names[i], color=colors[i],edgecolor='black')

    plt.legend(loc='best',ncol=1, shadow=False)
    plt.ylabel('(%)')
    # plt.title("Performance for " + dataset)
    plt.savefig(output_dir,bbox_inches='tight', dpi=200)

# plot_my_results("Images/adult"+ "_performance.png", "adult")
# plot_my_results("Images/bank"+ "_performance.png", "bank")
# plot_my_results("Images/compass"+ "_performance.png", "compass")
# plot_my_results("Images/kdd"+ "_performance.png", "kdd")


if __name__ == '__main__':
    # main("compass-gender")

    plot_my_results("Images/adult_single.png", "single_adult")
    plot_my_results("Images/bank_single.png", "single_bank")
    plot_my_results("Images/compass_single.png", "single_compass")
# plot_my_results("Images/kdd_single.png", "single_kdd")
